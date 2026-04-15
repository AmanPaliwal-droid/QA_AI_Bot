# 🔍 RAG Document Q&A Bot 

A Retrieval-Augmented Generation (RAG) based question-answering system that allows users to query a collection of documents and receive grounded, context-based answers with source citations. The system processes documents locally without relying on external APIs by using scikit-learn for embeddings and similarity search.

---

## Tech Stack

| Layer                   | Tool                  | Version |
| ----------------------- | --------------------- | ------- |
| Programming Language    | Python                | 3.9+    |
| Embeddings              | scikit-learn (TF-IDF) | latest  |
| Vector Search           | FAISS                 | 1.13.2  |
| Document Parsing (PDF)  | pdfplumber            | 0.11.0  |
| Document Parsing (DOCX) | python-docx           | 1.1.2   |
| Numerical Computing     | numpy                 | 1.26.4  |
| Environment Management  | python-dotenv         | 1.0.1   |
| UI                      | streamlit             | 1.35.0  |

---

## Architecture Overview

The system follows a standard RAG pipeline split into two stages:

### 1. Indexing Stage (Offline)

* Load documents from `/data`
* Extract clean text from PDF, TXT, and DOCX
* Split text into overlapping chunks
* Generate embeddings using TF-IDF (scikit-learn)
* Store vectors in FAISS index
* Save metadata (source file + page/section)

### 2. Query Stage (Online)

* Accept user query
* Convert query into TF-IDF embedding
* Perform similarity search in FAISS
* Retrieve top-K relevant chunks
* Generate answer strictly from retrieved context
* Return answer with source citations

---

## Chunking Strategy

**Fixed-size word chunking with overlap**

* Chunk size: 500 words
* Overlap: 80 words

### Why this approach?

* Ensures consistent chunk sizes for stable embeddings
* Prevents loss of context at boundaries
* Simpler and deterministic compared to NLP-based splitting

---

## Embedding Model and Vector Database

### Embedding Model: TF-IDF (scikit-learn)

* Fully local (no API dependency)
* Fast and lightweight
* Suitable for small to medium document collections

### Vector Database: FAISS

* Used for efficient similarity search
* Stores embeddings persistently on disk
* Uses cosine similarity (via normalized vectors)

### Why this setup?

* Avoids API costs and rate limits
* Works completely offline
* Good performance for assignment-scale datasets

---

## Document Collection

The `/data` folder contains the following documents:

| File                   | Type | Description                  |
| ---------------------- | ---- | ---------------------------- |
| bitcoin_whitepaper.pdf | PDF  | Cryptocurrency fundamentals  |
| solana.txt             | TXT  | Blockchain platform overview |
| aman_paliwal.txt       | TXT  | Trades Crypto currency       |
| ...                    | ...  | Additional documents         |

All documents are:

* At least 500+ words
* Non-trivial and meaningful

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-qa-bot
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add documents

Place your files inside `/data` folder.

### 5. Run indexing

```bash
python src/ingest.py
```

### 6. Run query bot

```bash
python src/query.py
```

---

## Environment Variables

No external API keys are required since the system runs fully locally.

---

## Example Queries

| Query                                            | Expected Answer Theme                |
| ------------------------------------------------ | ------------------------------------ |
| How does Bitcoin prevent double spending?        | Blockchain consensus, proof-of-work  |
| What is Solana known for?                        | High throughput, low fees            |
| What are key ideas in the Aman Paliwal document? | Document-specific insights           |
| Explain blockchain basics                        | Distributed ledger, decentralization |
| What is mentioned about scalability?             | Performance and transaction speed    |

---

## Sample Output

**Q:** How does Bitcoin prevent double spending?

**A:** Bitcoin prevents double spending using a distributed ledger system combined with proof-of-work consensus.

**Source:** bitcoin_whitepaper.pdf (Page 3)

---

## Known Limitations

| Limitation                                   | Reason                                     |
| -------------------------------------------- | ------------------------------------------ |
| Lower semantic accuracy than deep embeddings | TF-IDF does not capture contextual meaning |
| Struggles with paraphrased queries           | Keyword-based matching                     |
| No multi-hop reasoning                       | Retrieves independent chunks only          |
| PDF structure loss                           | Tables and formatting not preserved        |
| Requires re-indexing for new documents       | No incremental updates                     |

---

## Adjustable Parameters

| Parameter     | File      | Default |
| ------------- | --------- | ------- |
| CHUNK_SIZE    | ingest.py | 500     |
| CHUNK_OVERLAP | ingest.py | 80      |
| TOP_K         | query.py  | 5       |

---

## Design Decisions Summary

1. Used TF-IDF instead of API embeddings to ensure local execution
2. Selected FAISS for fast and scalable similarity search
3. Fixed-size chunking for predictable performance
4. Clear separation between indexing and querying for modularity
5. Context-restricted answering to avoid hallucination

---

## Handling Unknown Queries

If no relevant chunks are retrieved, the system responds with:

"I could not find an answer in the provided documents."

This ensures the model does not generate unsupported or hallucinated responses.
