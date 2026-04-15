"""
query.py — Interactive Q&A loop: embed query → retrieve chunks → generate answer.
Run: python src/query.py
"""

import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR       = Path("index")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL       = "gpt-4o-mini"   # swap to gpt-4o for higher quality
TOP_K           = 5                # configurable: number of chunks to retrieve

# ── load index ────────────────────────────────────────────────────────────────

def load_index():
    import faiss

    if not (INDEX_DIR / "vectors.faiss").exists():
        raise FileNotFoundError(
            "Index not found. Run  python src/ingest.py  first."
        )

    index  = faiss.read_index(str(INDEX_DIR / "vectors.faiss"))
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open(INDEX_DIR / "meta.json") as f:
        meta = json.load(f)

    return index, chunks, meta


# ── retrieval ─────────────────────────────────────────────────────────────────

def embed_query(query: str) -> np.ndarray:
    import faiss
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    vec = np.array([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(vec)
    return vec


def retrieve(query: str, index, chunks: List[Dict], top_k: int = TOP_K) -> List[Dict]:
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


# ── answer generation ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise document Q&A assistant. You answer questions ONLY based on the
provided context chunks. Each chunk is labelled with its source file and page number.

Rules:
- If the answer is present in the context, give a clear, concise answer and cite every
  source you used as [filename, p.N].
- If the answer is NOT found in the context, say exactly:
  "I could not find an answer to that question in the available documents."
  Do NOT use your training knowledge to fill gaps.
- Never fabricate information.
"""

def build_context_block(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[Chunk {i} | {c['source']} | p.{c['page']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(lines)


def generate_answer(query: str, chunks: List[Dict]) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    context = build_context_block(chunks)
    user_msg = f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,   # low temp → more factual
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ── display helpers ───────────────────────────────────────────────────────────

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
GREY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def print_banner(meta: Dict):
    sources = ", ".join(meta.get("sources", []))
    print(f"""
{BOLD}╔══════════════════════════════════════════════════╗
║          RAG Document Q&A Bot  🔍               ║
╚══════════════════════════════════════════════════╝{RESET}
  {GREY}Model : {meta.get('embedding_model')} + {LLM_MODEL}
  Chunks: {meta.get('num_chunks')}  |  Top-K: {TOP_K}
  Docs  : {sources}{RESET}

  Type your question and press Enter.
  Type {YELLOW}exit{RESET} or {YELLOW}quit{RESET} to stop.
  Type {YELLOW}sources{RESET} to list indexed documents.
""")


def print_answer(answer: str, chunks: List[Dict]):
    print(f"\n{GREEN}{BOLD}Answer:{RESET}")
    print(answer)

    print(f"\n{CYAN}{BOLD}Retrieved chunks:{RESET}")
    for i, c in enumerate(chunks, 1):
        score_bar = "█" * int(c["score"] * 10)
        print(
            f"  {GREY}[{i}] {c['source']}  p.{c['page']}  "
            f"score={c['score']:.3f} {score_bar}{RESET}"
        )
        preview = c["text"][:160].replace("\n", " ")
        print(f"      {GREY}…{preview}…{RESET}")
    print()


# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    print("\nLoading index…", end=" ", flush=True)
    index, chunks, meta = load_index()
    print("done.\n")

    print_banner(meta)

    while True:
        try:
            query = input(f"{BOLD}You ▶{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            print("Bye!")
            break

        if query.lower() == "sources":
            for s in sorted(meta.get("sources", [])):
                print(f"  • {s}")
            print()
            continue

        top_chunks = retrieve(query, index, chunks, top_k=TOP_K)
        answer     = generate_answer(query, top_chunks)
        print_answer(answer, top_chunks)


if __name__ == "__main__":
    main()
