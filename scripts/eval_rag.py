"""
Evaluate retrieval quality (Precision@k, MRR) for a simple RAG setup.

Run:
  python scripts/eval_rag.py --k 5

Optionally provide a dataset JSON with fields:
  [{"query": str, "relevant": ["doc1", "doc2", ...]}]

If no dataset is provided, uses a tiny built-in set and candidate corpus.
"""

from __future__ import annotations

import argparse
import json
from typing import List, Dict

import numpy as np

try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer
except Exception:
    raise SystemExit("Install requirements first (faiss-cpu, sentence-transformers).")


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    topk = retrieved[:k]
    hits = sum(1 for r in topk if r in relevant)
    return hits / max(k, 1)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / idx
    return 0.0


def build_index(docs: Dict[str, str], model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    ids = list(docs.keys())
    texts = [docs[i] for i in ids]
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return model, index, ids


def retrieve(model, index, ids: List[str], query: str, k: int) -> List[str]:
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, pos = index.search(q, k)
    return [ids[i] for i in pos[0] if 0 <= i < len(ids)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Path to JSON dataset")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    # Tiny default corpus and QA
    corpus = {
        "d1": "Retrieval-Augmented Generation grounds LLM outputs in retrieved documents.",
        "d2": "FAISS enables fast similarity search over dense vector embeddings.",
        "d3": "Chain-of-Thought prompting improves intermediate reasoning traces.",
        "d4": "Vector databases support payload filters and approximate nearest neighbor search.",
        "d5": "Transformers use attention to model token dependencies efficiently.
",
    }
    eval_set = [
        {"query": "How does RAG reduce hallucinations?", "relevant": ["d1"]},
        {"query": "What library speeds up dense similarity search?", "relevant": ["d2"]},
    ]

    if args.dataset:
        with open(args.dataset, "r", encoding="utf-8") as f:
            eval_set = json.load(f)

    model, index, ids = build_index(corpus)

    p_at_k, mrrs = [], []
    for ex in eval_set:
        retrieved = retrieve(model, index, ids, ex["query"], args.k)
        p_at_k.append(precision_at_k(retrieved, ex["relevant"], args.k))
        mrrs.append(mrr(retrieved, ex["relevant"]))

    print({
        "k": args.k,
        "precision@k": float(np.mean(p_at_k) if p_at_k else 0.0),
        "mrr": float(np.mean(mrrs) if mrrs else 0.0),
        "n_queries": len(eval_set),
    })


if __name__ == "__main__":
    main()
