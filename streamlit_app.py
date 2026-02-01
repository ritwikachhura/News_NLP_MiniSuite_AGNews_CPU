import os
# ---- Stability: limit native threads (helps prevent kernel/app crashes on CPU) ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

import numpy as np
import pandas as pd
import streamlit as st

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss

import umap
import hdbscan
from bertopic import BERTopic

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="News NLP Mini-Suite (AG News)", layout="wide")
st.title("📰 News NLP Mini‑Suite (AG News, CPU)")
st.caption("Semantic Search (SBERT + FAISS) • Topic Modeling (BERTopic) • Summarization (BART generate())")

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# -----------------------------
# Models / Data (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(n_docs: int = 5000):
    ds = load_dataset("ag_news")
    train = ds["train"].shuffle(seed=42).select(range(n_docs))
    docs = list(train["text"])     # ✅ convert to list
    labels = list(train["label"])  # ✅ convert to list
    return docs, labels

@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def load_summarizer_model():
    # generate()-based summarizer: no transformers.pipeline() needed
    model_id = "sshleifer/distilbart-cnn-12-6"   # CPU friendly summarization
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    mod.eval()
    torch.set_num_threads(1)
    return tok, mod, model_id

@st.cache_resource(show_spinner=True)
def build_faiss_index(docs, model_name: str = "all-MiniLM-L6-v2"):
    embedder = load_embedder(model_name)  # grab cached embedder inside
    emb = embedder.encode(
        docs,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return emb, index

@st.cache_resource(show_spinner=True)
def train_bertopic(docs, embeddings):
    # CPU-safe BERTopic setup
    umap_model = umap.UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=60,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="english",
        min_topic_size=60,
        calculate_probabilities=False,
        verbose=False
    )

    topics, _ = topic_model.fit_transform(docs, embeddings)
    return topic_model, topics

# -----------------------------
# Summarization helpers (generate())
# -----------------------------
def bart_summarize(tokenizer, model, text,
                  max_input_tokens=900,
                  max_new_tokens=90,
                  min_new_tokens=25,
                  num_beams=2):
    """
    CPU-friendly BART summarization via generate().
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            num_beams=num_beams,
            length_penalty=2.0,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_with_chunking(tokenizer, model, text, chunk_words=300):
    """
    Chunk long text into word chunks and summarize each.
    Optionally compress again if multiple chunks exist.
    """
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
    partial = [bart_summarize(tokenizer, model, c) for c in chunks]
    combined = " ".join(partial)

    if len(chunks) > 1:
        return bart_summarize(tokenizer, model, combined)
    return combined

# -----------------------------
# Semantic Search helper
# -----------------------------
def semantic_search(embedder, index, docs, labels, query, top_k=5):
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, top_k)

    rows = []
    for score, idx in zip(scores[0], ids[0]):
        rows.append({
            "score": float(score),
            "doc_id": int(idx),
            "label_id": int(labels[idx]),
            "label": LABEL_MAP.get(int(labels[idx]), str(labels[idx])),
            "snippet": docs[idx].replace("\n", " ")[:240] + "..."
        })
    return pd.DataFrame(rows)

# -----------------------------
# UI: Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    n_docs = st.slider("Number of documents", 1000, 10000, 5000, 1000)
    top_k = st.slider("Top‑K results", 3, 10, 5)
    train_topics = st.checkbox("Train topic model (BERTopic)", value=False)  # default OFF to reduce startup time
    st.divider()
    st.write("**Tip:** First run downloads models + dataset and may take a few minutes.")

# Load resources
docs, labels = load_data(n_docs)
embedder = load_embedder()
sum_tokenizer, sum_model, sum_model_id = load_summarizer_model()
embeddings, faiss_index = build_faiss_index(docs, model_name="all-MiniLM-L6-v2")

# Train topics (optional)
topic_model, topics = None, None
if train_topics:
    with st.spinner("Training BERTopic (CPU)..."):
        topic_model, topics = train_bertopic(docs, embeddings)
    info = topic_model.get_topic_info()
    st.sidebar.success(f"Topics ready: {(info.Topic != -1).sum()} (excluding outliers)")

# -----------------------------
# UI: Search + Results
# -----------------------------


# --- Preset queries near top of file ---
PRESET_QUERIES = [
    "AI technology company releases new product",
    "stock market falls after tech earnings",
    "government election and international conflict",
    "championship game ends in overtime thriller",
    "new smartphone launches with improved camera",
    "cybersecurity breach impacts major retailer",
    "central bank raises interest rates amid inflation",
    "space mission successfully launches new satellite",
]

# --- In the UI section ---
st.subheader("🔎 Semantic Search")

query_mode = st.radio(
    "Choose how to search:",
    ["Pick from presets", "Type my own"],
    horizontal=True
)

if query_mode == "Pick from presets":
    query = st.selectbox("Select a query", PRESET_QUERIES, index=0)
else:
    query = st.text_input("Enter your query", "AI technology company releases new product")

if st.button("Search", type="primary"):
    res = semantic_search(embedder, faiss_index, docs, labels, query, top_k=top_k)
    st.dataframe(res, use_container_width=True)

    st.subheader("🧩 Results: Topic + Summary")
    for _, row in res.iterrows():
        idx = int(row["doc_id"])
        title = f"Doc {idx} • score={row['score']:.3f} • {row['label']}"

        with st.expander(title):
            st.markdown("**Article**")
            st.write(docs[idx])

            if topic_model is not None:
                topic_id = int(topics[idx])
                if topic_id != -1:
                    words = topic_model.get_topic(topic_id)
                    st.markdown(f"**Topic {topic_id} keywords:** " + ", ".join([w for w, _ in words[:10]]))
                else:
                    st.markdown("**Topic:** outlier (-1)")

            with st.spinner("Summarizing (CPU)..."):
                summary = summarize_with_chunking(sum_tokenizer, sum_model, docs[idx])

            st.markdown("**Summary**")
            st.success(summary)
# st.subheader("🔎 Semantic Search")
# query = st.text_input("Enter a query", "AI technology company releases new product")

# if st.button("Search", type="primary"):
#     res = semantic_search(embedder, faiss_index, docs, labels, query, top_k=top_k)
#     st.dataframe(res, use_container_width=True)

#     st.subheader("🧩 Results: Topic + Summary")
#     for _, row in res.iterrows():
#         idx = int(row["doc_id"])   # ✅ use doc_id directly (no snippet matching!)
#         title = f"Doc {idx} • score={row['score']:.3f} • {row['label']}"

#         with st.expander(title):
#             st.markdown("**Article**")
#             st.write(docs[idx])

#             if topic_model is not None:
#                 topic_id = int(topics[idx])
#                 if topic_id != -1:
#                     words = topic_model.get_topic(topic_id)
#                     st.markdown(f"**Topic {topic_id} keywords:** " + ", ".join([w for w, _ in words[:10]]))
#                 else:
#                     st.markdown("**Topic:** outlier (-1)")

#             with st.spinner("Summarizing (CPU)..."):
#                 summary = summarize_with_chunking(sum_tokenizer, sum_model, docs[idx])

#             st.markdown("**Summary**")
#             st.success(summary)

# # -----------------------------
# # Optional topic overview
# # -----------------------------
st.divider()
st.subheader("📊 Topic Overview")
if topic_model is not None:
    st.write(topic_model.get_topic_info().head(15))
    st.caption("Tip: For visuals, use the notebook to export Plotly HTML charts.")
else:
    st.info("Enable 'Train topic model (BERTopic)' in the sidebar to view topic info.")