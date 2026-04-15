# 🔍 RAG Document Q&A Bot

A fully functional Retrieval-Augmented Generation (RAG) pipeline that lets you ask natural-language questions against a private document collection and receive grounded, cited answers. The system ingests PDF, TXT, and DOCX files; chunks, embeds, and indexes them in a persistent FAISS vector store; then retrieves the most relevant passages at query time and feeds them to an LLM for answer synthesis — with clear source citations.

---

## Tech Stack

| Layer | Tool | Version |
|---|---|---|
| Document loading (PDF) | `pdfplumber` | 0.11.0 |
| Document loading (DOCX) | `python-docx` | 1.1.2 |
| Embeddings | OpenAI `text-embedding-3-small` | via `openai` 1.30.5 |
| Vector database | `faiss-cpu` | 1.8.0 |
| Answer generation | OpenAI `gpt-4o-mini` | via `openai` 1.30.5 |
| Environment management | `python-dotenv` | 1.0.1 |
| Numerical compute | `numpy` | 1.26.4 |
| Web UI (optional) | `streamlit` | 1.35.0 |

Python 3.9 or later required.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INDEXING STEP                            │
│                    (run once: ingest.py)                        │
│                                                                 │
│  /data/*.pdf  ──┐                                               │
│  /data/*.txt  ──┤──▶ Text extraction ──▶ Word-overlap chunking  │
│  /data/*.docx ──┘         (pdfplumber/                          │
│                            python-docx)                         │
│                                  │                              │
│                                  ▼                              │
│                     Batched OpenAI embeddings                   │
│                   (text-embedding-3-small, dim=1536)            │
│                                  │                              │
│                                  ▼                              │
│                      FAISS IndexFlatIP (cosine)                 │
│                      persisted to /index/                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        QUERY STEP                               │
│               (interactive loop: query.py / app.py)             │
│                                                                 │
│  User question ──▶ Embed query ──▶ FAISS similarity search      │
│                                         │                       │
│                              top-K chunks + metadata            │
│                                         │                       │
│                                         ▼                       │
│                    GPT-4o-mini  (system prompt enforces          │
│                    context-only answering + citations)           │
│                                         │                       │
│                              Answer + [source, p.N] citations   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Chunking Strategy

**Strategy chosen: Fixed-size word chunking with overlap**

Each document page/section is split into windows of **500 words** with an **80-word overlap** between consecutive chunks.

**Why this strategy?**

- *Predictable token counts.* Embedding models have token limits; word-count windows guarantee no chunk ever exceeds the limit, unlike paragraph splitting where a single paragraph can be arbitrarily long.
- *Simplicity and reproducibility.* The logic is trivial to audit and produces the same output deterministically for the same input.
- *Overlap prevents context loss at boundaries.* An 80-word overlap means a sentence that straddles a chunk boundary is fully present in at least one of the two chunks, so retrieval won't miss it.

**Alternatives considered:**

| Strategy | Why rejected |
|---|---|
| Paragraph-based | PDF paragraph boundaries are unreliable (pdfplumber treats line-wrapped text as separate paragraphs); chunk sizes vary wildly from 10 to 800 words |
| Sentence-based (NLTK/spaCy) | Adds a heavy NLP dependency; chunks still need grouping to reach a useful size, making it a two-step process with no clear benefit for short factual documents |
| Recursive character splitting (LangChain) | Equivalent to word-overlap in practice; direct implementation is clearer and avoids a framework dependency |

---

## Embedding Model and Vector Database

### Embedding model: `text-embedding-3-small` (OpenAI)

- **Dimension**: 1 536
- **Context window**: 8 191 tokens — comfortable for 500-word chunks
- **Cost**: ~$0.02 per million tokens — negligible for a 5-document corpus
- **Quality**: Outperforms older `ada-002` on MTEB benchmarks while being cheaper
- **Why not a local model?** Models like `all-MiniLM-L6-v2` are fine for demos but underperform on domain-diverse corpora; the assignment permits API usage and the cost difference is trivial

### Vector database: FAISS (`IndexFlatIP`, cosine similarity)

- **Persistence**: index written to `/index/vectors.faiss` — no re-indexing on restart
- **Separation**: `ingest.py` (write path) and `query.py` (read path) are completely separate scripts with no shared in-memory state
- **Why FAISS over ChromaDB/Qdrant?** For a corpus of 4–5 documents (~200–500 chunks), an exact flat index is fast and eliminates the overhead of a client-server database. ChromaDB would be a better choice if the corpus grew to millions of vectors.
- Vectors are L2-normalised before insertion so that `IndexFlatIP` (inner product) becomes equivalent to cosine similarity — the standard metric for semantic search.

---

## Document Collection

Place your documents in the `/data` folder. Suggested documents (publicly available):

| File | Description |
|---|---|
| `attention_is_all_you_need.pdf` | Transformer architecture paper (Vaswani et al., 2017) |
| `bitcoin_whitepaper.pdf` | Bitcoin: A Peer-to-Peer Electronic Cash System (Satoshi Nakamoto) |
| `climate_change_ipcc_summary.txt` | IPCC AR6 Summary for Policymakers (plain text) |
| `gdpr_regulation.txt` | GDPR key articles (plain text excerpt) |
| `python_pep8_style_guide.txt` | PEP 8 — Style Guide for Python Code |

At least one must be a PDF. All documents should be at least 500 words.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/rag-qa-bot.git
cd rag-qa-bot
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-...your-key-here...
```

### 5. Add documents to /data

```bash
# Example: download the Bitcoin whitepaper
curl -o data/bitcoin_whitepaper.pdf https://bitcoin.org/bitcoin.pdf
```

Add at least 4–5 documents of your choice (PDF/TXT/DOCX, min 500 words each).

### 6. Run the ingestion pipeline

```bash
python src/ingest.py
```

You will see progress for each stage: loading → chunking → embedding → indexing. This step only needs to run once (or whenever you add new documents).

### 7a. Run the CLI Q&A bot

```bash
python src/query.py
```

Type your questions at the prompt. Type `exit` to quit.

### 7b. Run the Streamlit web UI (optional)

```bash
streamlit run src/app.py
```

Open `http://localhost:8501` in your browser.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | OpenAI API key for embeddings and GPT answer generation |

**Never commit your actual `.env` file.** It is listed in `.gitignore`.

---

## Project Structure

```
rag-qa-bot/
├── data/                   # Your documents (PDF, TXT, DOCX)
├── index/                  # Generated by ingest.py — gitignored
│   ├── vectors.faiss       # FAISS binary index
│   ├── chunks.pkl          # Chunk text + metadata
│   └── meta.json           # Index statistics
├── src/
│   ├── ingest.py           # Indexing pipeline (run once)
│   ├── query.py            # CLI interactive Q&A loop
│   └── app.py              # Optional Streamlit web UI
├── .env.example            # Template for environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Example Queries

The following queries assume the suggested document set above.

| Query | Expected answer theme |
|---|---|
| "What problem does the Transformer architecture solve compared to RNNs?" | Parallelisation, long-range dependencies, attention mechanism |
| "How does Bitcoin prevent double-spending without a central authority?" | Proof-of-work, longest chain rule, distributed consensus |
| "What are the key rights granted to individuals under GDPR?" | Right to erasure, data portability, consent, access |
| "What does PEP 8 say about maximum line length?" | 79 characters for code, 72 for docstrings |
| "What temperature increase does the IPCC consider dangerous?" | 1.5 °C threshold, 2 °C upper bound |
| "Who invented the internet?" | *(not in documents)* → bot responds: "I could not find an answer…" |

---

## Known Limitations

| Limitation | Reason |
|---|---|
| Tables and figures in PDFs are lost | `pdfplumber` extracts flowing text; structured table data requires `camelot` or `tabula` |
| Multi-document synthesis is limited | Retrieved chunks are ranked individually; cross-document reasoning depends on all relevant chunks landing in the top-K |
| Fixed chunk size ignores semantic boundaries | A 500-word window may split a key definition across two chunks, reducing retrieval precision |
| No query expansion or HyDE | The query is embedded as-is; hypothetical document embedding (HyDE) or synonym expansion could improve recall |
| Index must be rebuilt to add new documents | Incremental indexing (checking which files are new) is not yet implemented |
| English only | The embedding model works for other languages but the system prompt and UI are English-only |

---

## Adjustable Parameters

Edit the constants at the top of each script:

| Parameter | File | Default | Effect |
|---|---|---|---|
| `CHUNK_SIZE` | `ingest.py` | 500 | Words per chunk |
| `CHUNK_OVERLAP` | `ingest.py` | 80 | Overlap words between chunks |
| `EMBEDDING_MODEL` | `ingest.py` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `query.py` / `app.py` | `gpt-4o-mini` | Generation model |
| `TOP_K` | `query.py` / `app.py` | 5 | Chunks retrieved per query |

---

## Architecture Decision Summary

1. **Chunking**: Fixed-size word windows (500 w / 80 w overlap) — predictable token budget, no extra dependencies.
2. **Embeddings**: OpenAI `text-embedding-3-small` — best price/performance ratio; 1 536-dim vectors suitable for cosine search.
3. **Vector DB**: FAISS flat index (cosine via normalised inner product) — zero-dependency, persists to a single binary file, exact search at this corpus scale.
4. **LLM**: `gpt-4o-mini` — cost-effective with strong instruction-following; `temperature=0.1` and a strict system prompt prevent hallucination.
5. **Context-only answering**: The system prompt explicitly forbids the model from drawing on training knowledge if the answer isn't in the retrieved chunks.
