---
title: Embeddings & Vector Stores
summary: How embeddings represent data, vector search, ANN, RAG, and operational considerations.
---

# Embeddings & Vector Stores

This chapter explains embeddings (text, image, multimodal), how to build and evaluate them, and how vector stores and ANN indexes enable retrieval at scale. It includes practical notes for building RAG systems and operationalizing vector databases.

## 1. Why embeddings matter

- Embeddings convert heterogeneous data (text, images, audio, structured features) into dense vectors in a shared space where semantic similarity becomes geometric proximity.
- Use-cases: semantic search, recommendations, reranking, clustering, anomaly detection, and retrieval-augmented generation (RAG).

## 2. Embedding techniques and models

- Classic: Word2Vec, GloVe, FastText (subword support).
- Contextual / sentence-level: BERT, Sentence-BERT, T5-based encoders — better for sentence/document similarity.
- Multimodal: CLIP, BLIP, and other joint text-image models map different modalities into a joint vector space.
- Structured/graph: DeepWalk, Node2Vec, GraphSAGE embed graph nodes for link prediction and classification.

Choose an embedding model based on domain (short queries vs long documents), latency, and cost.

## 3. Evaluating embeddings

- Retrieval metrics: precision@k, recall@k, NDCG (position-aware ranking quality).
- Benchmarks: BEIR and MTEB for retrieval and cross-task evaluation.
- Practical workflow: create a small held-out test set, measure top-k metrics, and inspect qualitative failures.

## 4. Training and fine-tuning embeddings

- Dual-encoder / contrastive learning: train encoders so positive pairs are close and negatives are far (InfoNCE loss or margin-based losses).
- Data strategies: mining hard negatives, synthetic augmentations, and domain-specific fine-tuning.
- Parameter-efficient adaptation: freeze base encoders and fine-tune small adapters when data is limited.

## 5. Retrieval-Augmented Generation (RAG)

- RAG pipeline: document chunking → embed chunks → index in a vector store → at query time embed query → ANN search → build prompt with retrieved context → answer with LLM.
- RAG reduces hallucinations by grounding generation in retrieved facts. Chunk-size, overlap, and embedding choice strongly affect relevance.

Basic RAG pseudocode:

```python
# 1) Build embeddings and index
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')
docs = ["...document text...", ...]
embs = model.encode(docs, convert_to_numpy=True)

index = faiss.IndexFlatIP(embs.shape[1])
faiss.normalize_L2(embs)
index.add(embs)

# 2) Query
q = "What causes embedding drift?"
q_emb = model.encode([q], convert_to_numpy=True)
faiss.normalize_L2(q_emb)
_, ids = index.search(q_emb, k=5)
retrieved = [docs[i] for i in ids[0]]

# 3) Construct prompt (prepend retrieved) and call LLM
prompt = """
Context:
%s

Question: %s
""" % ("\n---\n".join(retrieved), q)

# send `prompt` to your LLM provider
```

## 6. Vector search & ANN algorithms

- Exact NN: precise but slow at scale.
- ANN families: LSH, tree-based (KD-tree), graph-based (HNSW), and optimized kernels (ScaNN). HNSW (and its HNSWlib implementations) is a practical default for many workloads.
- Hybrid search: combine keyword filters (BM25) with vector re-ranking to handle rare / factual tokens.

## 7. Vector databases and operational considerations

- Examples: Pinecone, Weaviate, Chroma, Milvus, Qdrant, FAISS (library). Cloud offerings: Vertex AI Vector Search, Amazon OpenSearch with vector plugin.
- Operational concerns: sharding, replication, index rebuilds, reindexing on model upgrades, latency SLAs, authorization, and cost.
- Embedding drift: plan reindexing cadence and version embeddings (store model id + embedding schema in metadata).

## 8. Best practices

- Choose embedding model by retrieval task: small, fast models for low-latency, stronger models for high-quality relevance.
- Normalize and store metadata alongside vectors for filtering and provenance.
- Monitor drift: track retrieval quality over time and automate sample audits.
- Use dense + sparse hybrid retrieval for robust results.

## 9. Hands-on exercises

1. Build a small FAISS index with `all-MiniLM-L6-v2` and measure precision@5 on a held-out Q&A set.
2. Implement a simple RAG loop using local FAISS and an LLM API; compare results when you change chunk size.
3. Evaluate a vector DB (e.g., Chroma vs FAISS) for your workload and document latency and cost trade-offs.

## References and further reading

- See the detailed notes and whitepaper in `quick-knowledge/2. Embeddings and Vector Stores/` for deep dives and the source slides.
- Libraries: `sentence-transformers`, `faiss`, `hnswlib`, `langchain`, `pinecone-client`.

---

If you'd like, I can also add a tiny `examples/embeddings/` folder with runnable scripts and a `requirements.txt` to let you test locally — should I create that next? 
