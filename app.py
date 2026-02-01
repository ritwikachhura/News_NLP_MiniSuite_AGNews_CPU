"""News NLP demo app (refactored)

Features:
- Load AG News dataset (subset configurable)
- Compute / cache SBERT embeddings and build a FAISS index
- Semantic search with direct doc indices
- Topic modeling via BERTopic
- Collapsible summarization using a small BART model with chunking

Run: python app_refactored.py --help
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Iterable, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import umap
import hdbscan


# ---- Configuration ----
DEFAULT_SAVED_EMB = Path("agnews_sbert_embeddings.npy")
SBERT_MODEL = "all-MiniLM-L6-v2"
SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"
SEED = 42


# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def set_cpu_env() -> None:
    """Set environment variables to reduce nested parallelism on some platforms."""
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"


def load_ag_news(n: int) -> Tuple[List[str], List[int]]:
    ds = load_dataset("ag_news")
    train = ds["train"].shuffle(seed=SEED).select(range(n))
    return train["text"], train["label"]


def get_embeddings(
    docs: Iterable[str],
    model_name: str = SBERT_MODEL,
    cache_path: Path = DEFAULT_SAVED_EMB,
    batch_size: int = 64,
    force: bool = False,
) -> np.ndarray:
    """Compute or load cached embeddings (float32, normalized)"""
    if cache_path.exists() and not force:
        logger.info("Loading embeddings from %s", cache_path)
        return np.load(cache_path)

    logger.info("Computing embeddings with %s (this may take a while)", model_name)
    model = SentenceTransformer(model_name)
    arr = model.encode(list(docs), batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(arr, dtype="float32")
    np.save(cache_path, emb)
    logger.info("Saved embeddings to %s", cache_path)
    return emb


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("Built FAISS index (%d vectors, dim=%d)", index.ntotal, dim)
    return index


def semantic_search(
    index: faiss.IndexFlatIP,
    embedder: SentenceTransformer,
    docs: List[str],
    labels: List[int],
    query: str,
    top_k: int = 5,
) -> pd.DataFrame:
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, top_k)
    rows = []
    for score, idx in zip(scores[0], ids[0]):
        rows.append(
            {
                "score": float(score),
                "label": int(labels[idx]),
                "doc_id": int(idx),
                "text": docs[idx].replace("\n", " ")[:400] + "...",
            }
        )
    return pd.DataFrame(rows)


def compute_recall_at_k(index: faiss.IndexFlatIP, embeddings: np.ndarray, labels: List[int], k: int = 5, n_queries: int = 200) -> float:
    n_queries = min(n_queries, embeddings.shape[0])
    hits = 0
    for i in range(n_queries):
        q = embeddings[i : i + 1]
        scores, ids = index.search(q, k + 1)
        retrieved = [j for j in ids[0] if j != i][:k]
        if any(labels[j] == labels[i] for j in retrieved):
            hits += 1
    recall = hits / n_queries
    logger.info("Recall@%d = %.4f (n=%d)", k, recall, n_queries)
    return recall


class Summarizer:
    """Lazy wrapper for a summarization model."""

    def __init__(self, model_id: str = SUMMARY_MODEL):
        self.model_id = model_id
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSeq2SeqLM] = None

    def _ensure_loaded(self) -> None:
        if self._model is None or self._tokenizer is None:
            logger.info("Loading summarization model: %s", self.model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            self._model.eval() # type: ignore
            torch.set_num_threads(1)

    def summarize(self, text: str, max_input_tokens: int = 900, **gen_kwargs) -> str:
        self._ensure_loaded()
        tok = self._tokenizer(text, truncation=True, return_tensors="pt", max_length=max_input_tokens) # type: ignore
        with torch.no_grad():
            ids = self._model.generate(**tok, **gen_kwargs) # type: ignore
        return self._tokenizer.decode(ids[0], skip_special_tokens=True) # type: ignore

    def chunked(self, text: str, chunk_words: int = 300, **gen_kwargs) -> str:
        words = text.split()
        chunks = [" ".join(words[i : i + chunk_words]) for i in range(0, len(words), chunk_words)]
        partials = [self.summarize(c, **gen_kwargs) for c in chunks]
        combined = " ".join(partials)
        if len(chunks) > 1:
            # compress multiple partial summaries
            return self.summarize(combined, max_input_tokens=900, **gen_kwargs)
        return combined


def run_pipeline(subset: int = 5000, cache_path: Path = DEFAULT_SAVED_EMB) -> None:
    set_cpu_env()
    np.random.seed(SEED)

    docs, labels = load_ag_news(subset)
    logger.info("Loaded %d docs", len(docs))

    embeddings = get_embeddings(docs, cache_path=cache_path)
    faiss_index = build_faiss_index(embeddings)

    # use a single embedder instance for both indexing and searching
    embedder = SentenceTransformer(SBERT_MODEL)

    # Quick evaluation
    compute_recall_at_k(faiss_index, embeddings, labels, k=5, n_queries=200)
    compute_recall_at_k(faiss_index, embeddings, labels, k=10, n_queries=200)

    # Topic modeling (may take a while)
    umap_model = umap.UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED, low_memory=True)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=60, metric="euclidean", cluster_selection_method="eom", prediction_data=False)

    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, language="english", calculate_probabilities=False, verbose=False)
    topics, _ = topic_model.fit_transform(docs, embeddings)
    logger.info("BERTopic discovered %d topics", topic_model.get_topic_info().shape[0])

    # Example semantic search + summarize
    q = "AI technology company releases new product"
    results = semantic_search(faiss_index, embedder, docs, labels, q, top_k=3)

    summarizer = Summarizer()
    enriched_rows = []
    for row in results.to_dict(orient="records"):
        idx = int(row["doc_id"])
        topic_id = int(topics[idx])
        topic_words = topic_model.get_topic(topic_id) if topic_id != -1 else []
        summary = summarizer.chunked(docs[idx], chunk_words=300, max_new_tokens=120, min_new_tokens=35, num_beams=4, length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True)
        enriched_rows.append({**row, "topic": topic_id, "topic_words": ", ".join([w for w, _ in topic_words[:6]]) if topic_words else "outlier", "summary": summary})

    logger.info("Assistant results:\n%s", pd.DataFrame(enriched_rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="News NLP demo")
    p.add_argument("--subset", type=int, default=5000, help="Number of training docs to use (default: 5000)")
    p.add_argument("--force-emb", action="store_true", help="Recompute embeddings even if cache exists")
    p.add_argument("--cache", type=Path, default=DEFAULT_SAVED_EMB, help="Embedding cache path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subset=args.subset, cache_path=args.cache)