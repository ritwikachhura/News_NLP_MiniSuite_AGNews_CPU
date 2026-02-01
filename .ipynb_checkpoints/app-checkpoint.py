import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
from bertopic import BERTopic
from transformers import pipeline
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # prevents nested parallelism


np.random.seed(42)


#load data set 
df = load_dataset('ag_news')

# CPU-friendly subset (increase later to 10k/20k)N = 5000
N=5000
train = df['train'].shuffle(seed = 42).select(range(N))

docs =train['text']
labels=train['label']

print("Docs Count:", len(docs))
print("\nSample:\n", docs[0][:500])

lengths = [len(d.split()) for d in docs]
pd.Series(lengths).describe()


plt.figure(figsize=(7,4))
plt.hist(lengths, bins=40)
plt.title("AG News Article Lengths (words)")
plt.xlabel("Words")
plt.ylabel("Count")
plt.show()


#semantic search 
EMB_PATH = "agnews_sbert_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"

STModel = SentenceTransformer(MODEL_NAME)

if os.path.exists(EMB_PATH):
    embeddings = np.load(EMB_PATH)
    print("Loaded cached embeddings:", embeddings.shape)
else:    
    t0 = time.time()    
    embeddings = STModel.encode(
        docs,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )    
    embeddings = np.asarray(embeddings, dtype="float32")
    np.save(EMB_PATH, embeddings)
    print("Computed embeddings:", embeddings.shape, "time:", round(time.time()-t0, 2), "sec")

#FAISS index

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print("FAISS index size:", index.ntotal)

# IMPORTANT: embedder must be SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query, top_k=5):
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, top_k)

    out = []
    for score, idx in zip(scores[0], ids[0]):
        out.append({
            "score": float(score),
            "label": int(labels[idx]),
            "doc_id": int(idx),
            "text": docs[idx].replace("\n", " ")[:400] + "..."
        })
    return pd.DataFrame(out)

semantic_search("stock market falls after tech earnings", top_k=5)

#Eval

def recall_at_k(k=5, n_queries=200):
    hits = 0
    for i in range(n_queries):
        # use the doc itself as the query (self-retrieval test)
        q = embeddings[i:i+1]
        scores, ids = index.search(q, k+1)  # includes itself
        retrieved = [j for j in ids[0] if j != i][:k]
        if any(labels[j] == labels[i] for j in retrieved):
            hits += 1
    return hits / n_queries

print("Recall@5:", recall_at_k(k=5, n_queries=200))
print("Recall@10:", recall_at_k(k=10, n_queries=200))


#This reduces memory + makes clustering less fragile.

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
    calculate_probabilities=False,   # important for CPU memory
    verbose=True
)

topics, _ = topic_model.fit_transform(docs, embeddings)
print(topic_model.get_topic_info().head(10))
topic_info = topic_model.get_topic_info()
print(topic_info.head(15))

#inspecting Topics and Representative Docs 
# pick first non-outlier topic (topic -1 = outliers)
non_outliers = topic_info[topic_info["Topic"] != -1]
t = int(non_outliers["Topic"].iloc[0])

print("Topic:", t)
topic_words = topic_model.get_topic(t)
print(f"Topic {t} words:")
for word, weight in topic_words:
    print(f"{word}: {weight:.4f}")

print("\nRepresentative doc:\n", topic_model.get_representative_docs(t)[0][:700])


#reduce topics for readability
topic_model.reduce_topics(docs, nr_topics=15)
topic_model.get_topic_info().head(20)

#Visualizing now
fig = topic_model.visualize_barchart(top_n_topics=10)
fig.show()

#Summarizing

MODEL_ID = "sshleifer/distilbart-cnn-12-6"  # fast-ish on CPU; can also try facebook/bart-large-cnn

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#bypass transformer reduced registry
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

# CPU settings
model.eval()
torch.set_num_threads(1)  # helps stability on some setups

def bart_summarize(text, max_input_tokens=900, max_new_tokens=120, min_new_tokens=35):
    # Truncate to avoid exceeding model max length
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            num_beams=4,
            length_penalty=2.0,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(bart_summarize(docs[0]))

#chunking summarizer
def summarize_with_chunking(text, chunk_words=300, **kwargs):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_words]) for i in range(0, len(words), chunk_words)]

    partial = [bart_summarize(c, **kwargs) for c in chunks]
    combined = " ".join(partial)

    # If multiple chunks, compress again
    if len(chunks) > 1:
        return bart_summarize(combined, max_input_tokens=900, max_new_tokens=120, min_new_tokens=35)

    return combined

print(summarize_with_chunking(docs[0]))

#demo
for i in [0, 10, 25]:
    print("="*100)
    print("ORIGINAL:\n", docs[i][:700], "\n")
    print("SUMMARY:\n", summarize_with_chunking(docs[i]))


#news workflow
#one search model
def newsroom_assistant(query, top_k=3):
    results = semantic_search(query, top_k=top_k)

    enriched = []
    for _, row in results.iterrows():
        # find original doc index by matching snippet (simple approach for demo)
        # better: in search function return idx; here we’ll do quick lookup
        snippet = row["text"].replace("...", "")
        idx = next((i for i, d in enumerate(docs) if snippet[:80] in d), None)

        if idx is None:
            enriched.append({**row, "topic": None, "topic_words": None, "summary": None})
            continue

        topic_id = topics[idx]
        topic_words = topic_model.get_topic(topic_id) if topic_id != -1 else []

        summary = summarize_with_chunking(docs[idx])

        enriched.append({
            **row,
            "topic": int(topic_id),
            "topic_words": ", ".join([w for w, _ in topic_words[:6]]) if topic_words else "outlier",
            "summary": summary
        })

    return pd.DataFrame(enriched)

newsroom_assistant("AI technology company releases new product", top_k=3)