"""
ingest.py — Document ingestion, chunking, embedding, and vector store creation.
Run this once before querying: python src/ingest.py
"""

import os
import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
INDEX_DIR   = Path("index")
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 80

# ── helpers ───────────────────────────────────────────────────────────────────

def load_documents() -> List[Dict[str, Any]]:
    docs = []

    # PDF
    try:
        import pdfplumber
        for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
            print(f"  📄 Loading PDF: {pdf_path.name}")
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    text = text.strip()
                    if len(text) > 50:
                        docs.append({
                            "text": text,
                            "source": pdf_path.name,
                            "page": page_num,
                        })
    except ImportError:
        print("  ⚠️  pdfplumber not installed — skipping PDFs")

    # TXT
    for txt_path in sorted(DATA_DIR.glob("*.txt")):
        print(f"  📄 Loading TXT: {txt_path.name}")
        text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        docs.append({"text": text, "source": txt_path.name, "page": 1})

    # DOCX
    try:
        import docx
        for docx_path in sorted(DATA_DIR.glob("*.docx")):
            print(f"  📄 Loading DOCX: {docx_path.name}")
            document = docx.Document(str(docx_path))
            full_text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
            docs.append({"text": full_text, "source": docx_path.name, "page": 1})
    except ImportError:
        print("  ⚠️  python-docx not installed — skipping DOCX")

    return docs


def chunk_document(doc: Dict[str, Any], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    words = doc["text"].split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text,
            "source": doc["source"],
            "page": doc["page"],
            "chunk_index": len(chunks),
        })

        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


# 🔥 UPDATED: LOCAL EMBEDDINGS (NO OPENAI)
def embed_chunks(chunks):
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("  🤖 Using lightweight TF-IDF embeddings...")

    texts = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer(max_features=5000)
    embeddings = vectorizer.fit_transform(texts).toarray()

    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray):
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def save_index(index, chunks: List[Dict[str, Any]]):
    INDEX_DIR.mkdir(exist_ok=True)

    import faiss
    faiss.write_index(index, str(INDEX_DIR / "vectors.faiss"))

    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    meta = {
        "num_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "sources": list({c["source"] for c in chunks}),
    }

    with open(INDEX_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✅ Saved {len(chunks)} chunks to {INDEX_DIR}/")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n═══ RAG Ingestion Pipeline ═══\n")

    print("① Loading documents…")
    docs = load_documents()
    if not docs:
        print("  ❌ No documents found in /data — add some files first.")
        return

    print(f"  Loaded {len(docs)} sections from {len({d['source'] for d in docs})} files.\n")

    print("② Chunking…")
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']} p.{doc['page']} → {len(chunks)} chunks")

    print(f"  Total: {len(all_chunks)} chunks\n")

    print("③ Embedding…")
    embeddings = embed_chunks(all_chunks)
    print(f"  Embedding shape: {embeddings.shape}\n")

    print("④ Building FAISS index & saving…")
    index = build_faiss_index(embeddings)
    save_index(index, all_chunks)

    print("\n🎉 Ingestion complete. Now run Streamlit app.\n")


if __name__ == "__main__":
    main()