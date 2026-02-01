# 📰 News NLP Mini‑Suite (AG News • CPU)

**Semantic Search • Topic Modeling • Summarization + Streamlit Demo**

This repository is a **portfolio-ready NLP mini-suite** built on real news text (**AG News**). It demonstrates an end‑to‑end content workflow often used in newsroom/content platforms:

✅ **Semantic Search** (Sentence‑BERT + FAISS)  
✅ **Topic Modeling** (BERTopic with CPU‑safe settings)  
✅ **Summarization** (HuggingFace Seq2Seq using `generate()` — no `pipeline()` required)  
✅ **Streamlit App** for interactive exploration

***

## ✨ Why this project is useful

News/content teams often need to:

*   find related stories quickly (semantic search),
*   understand content themes (topic modeling),
*   generate short briefs (summarization).

This repo implements all three in a **CPU-friendly**, demo‑ready format.

***

## 📁 Project Structure

```text
.
├── News_NLP_MiniSuite_AGNews_CPU.ipynb   # Main notebook (all modules + explanation)
├── app.py                               # Streamlit demo app
├── requirements.txt                     # Dependencies
└── README.md                            # You are here
```

***

## 🧠 Features

### 1) 🔎 Semantic Search (Sentence‑BERT + FAISS)

*   Uses **Sentence‑BERT** (`all-MiniLM-L6-v2`) to embed news articles.
*   Builds a **FAISS cosine similarity index** (inner product on normalized vectors).
*   Lets users search with natural language queries:
    *   “AI technology company releases new product”
    *   “stock market falls after tech earnings”
*   Returns top‑K results with similarity scores, snippets, and doc IDs.

***

### 2) 🧩 Topic Modeling (BERTopic)

*   Uses BERTopic to cluster articles into topics and extract topic keywords.
*   Includes **CPU‑safe settings** to reduce kernel/app crashes:
    *   thread limiting (`OMP_NUM_THREADS=1` etc.)
    *   UMAP `low_memory=True`
    *   HDBSCAN `prediction_data=False`
    *   `calculate_probabilities=False`
*   Outputs:
    *   topic table (topic sizes)
    *   topic keywords
    *   representative documents
    *   optional Plotly HTML export for visuals

***

### 3) ✨ Summarization (generate()-based, no pipelines)

Some environments don’t support the Transformers `pipeline("summarization")` or `"text2text-generation"` tasks.  
To keep this project robust, summarization uses:

*   `AutoTokenizer + AutoModelForSeq2SeqLM`
*   `model.generate()` (beam search, repetition control)
*   Optional chunking for longer text

Model used (CPU-friendly):

*   `sshleifer/distilbart-cnn-12-6`

> ⚠️ Note: Abstractive summarization can hallucinate. For high‑risk domains, consider extractive summaries or grounding/verification.

***

## 🚀 Quickstart

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2A) Run the notebook

```bash
jupyter notebook
```

Open:

*   `News_NLP_MiniSuite_AGNews_CPU.ipynb`

### 2B) Run the Streamlit demo

```bash
streamlit run streamlit_app.py
```

***

## 🖥️ Streamlit Demo (What you can do)

The app supports:

*   **Preset query dropdown** (demo-friendly)
*   optional **custom query**
*   semantic search results table
*   expandable results showing:
    *   full article text
    *   optional topic keywords (if BERTopic enabled)
    *   generated summary (CPU)

Sidebar controls:

*   number of documents (1k–10k)
*   top‑K retrieval
*   toggle BERTopic training (turn off for faster load)

***

## 🧪 Example Queries

Use these in the app or notebook:

*   AI / Tech: `AI technology company releases new product`
*   Markets: `stock market falls after tech earnings`
*   Politics/World: `government election and international conflict`
*   Sports: `championship game ends in overtime thriller`
*   Security: `cybersecurity breach impacts major retailer`

***

## 📊 APP Functionality Snapshots

1.  Semantic search results table:
  <img width="1478" height="460" alt="image" src="https://github.com/user-attachments/assets/560cc08e-a2c2-4f19-9743-752e603730dc" />

2.  Topic table (`topic_model.get_topic_info()`)
  <img width="1450" height="197" alt="image" src="https://github.com/user-attachments/assets/73586995-0947-4284-8c72-316b1c14a3ef" />

3.  Topic keyword output (per topic)
  <img width="1448" height="499" alt="image" src="https://github.com/user-attachments/assets/757f3f34-f5a3-4a86-83b5-b9d751c67f71" />

4.  Original vs summary output
  <img width="1467" height="300" alt="image" src="https://github.com/user-attachments/assets/a7997651-bbc7-4848-9ec3-1cf95fb1fe9c" />

5.  Streamlit UI: query + expanded results
  <img width="1472" height="184" alt="image" src="https://github.com/user-attachments/assets/11a03bfa-2b5c-4ffe-a880-5f7b10291bb3" />


***

## ⚙️ Notes on Performance (CPU)

*   First run downloads dataset/models and may take a few minutes.
*   BERTopic is heavier than search/summarization on CPU.
    *   Start with 5,000 docs and scale up if memory allows.
*   Thread limiting is included to reduce instability in constrained environments.

***

## 🔧 Common Issues & Fixes

### Streamlit caching errors (UnhashableParamError)

HuggingFace dataset columns and model objects aren’t hashable.  
Fixes used in this repo:

*   Convert dataset columns to lists (`list(train["text"])`)
*   Avoid passing unhashable objects into cached functions (or prefix args with `_`)
