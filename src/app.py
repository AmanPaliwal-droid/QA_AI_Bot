"""
app.py — Streamlit web UI for LOCAL RAG Q&A bot (NO OpenAI).
Run: streamlit run src/app.py
"""

import pickle
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import streamlit as st

INDEX_DIR = Path("index")
TOP_K = 5

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Q&A Bot (Local)",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
  .main { background: #0f1117; color: #e8e8e8; }

  .answer-box {
    background: #1a1d27;
    border-left: 4px solid #4ade80;
    border-radius: 6px;
    padding: 1rem;
    margin: 1rem 0;
  }

  .chunk-card {
    background: #1e2132;
    border-radius: 6px;
    padding: 0.7rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
  }
</style>
""", unsafe_allow_html=True)

# ── load index ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_index():
    import faiss
    index = faiss.read_index(str(INDEX_DIR / "vectors.faiss"))

    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open(INDEX_DIR / "meta.json") as f:
        meta = json.load(f)

    return index, chunks, meta


# ── retrieval (NO embeddings API) ─────────────────────────────────────────────

def embed_query(query: str):
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query]).toarray().astype("float32")

    return query_vec


def retrieve(query: str, index, chunks: List[Dict], top_k: int):
    query_vec = embed_query(query)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)

    return results


# ── answer generation (LOCAL) ─────────────────────────────────────────────────

def generate_answer(query: str, retrieved_chunks: List[Dict]):
    if not retrieved_chunks:
        return "No relevant information found."

    answer_parts = []

    for c in retrieved_chunks:
        answer_parts.append(
            f"{c['text']}\n\n(Source: {c['source']} p.{c['page']})"
        )

    return "\n\n---\n\n".join(answer_parts)


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🔍 Local RAG Q&A Bot (No API)")

if not (INDEX_DIR / "vectors.faiss").exists():
    st.error("Index not found. Run `python src/ingest.py` first.")
    st.stop()

index, chunks, meta = load_index()

# Sidebar
with st.sidebar:
    st.header("📊 Info")
    st.metric("Chunks", meta.get("num_chunks"))
    st.metric("Mode", "Fully Local (No API)")
    st.divider()

    st.markdown("**Documents:**")
    for s in sorted(meta.get("sources", [])):
        st.markdown(f"- {s}")

    top_k = st.slider("Top-K", 1, 10, TOP_K)

# Chat memory
if "history" not in st.session_state:
    st.session_state.history = []

# Show history
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.markdown(f'<div class="answer-box">{turn["answer"]}</div>', unsafe_allow_html=True)

# Input
query = st.chat_input("Ask something from your documents...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            top_chunks = retrieve(query, index, chunks, top_k)
            answer = generate_answer(query, top_chunks)

        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    st.session_state.history.append({
        "query": query,
        "answer": answer,
    })