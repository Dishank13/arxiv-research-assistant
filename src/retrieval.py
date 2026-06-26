"""
Phase 2: Hybrid Retrieval Module
==================================
Dense search (Qdrant), sparse search (BM25), Reciprocal Rank Fusion,
cross-encoder reranking, and combined search pipelines.
"""

import os
import pickle
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

load_dotenv()

# ─── Lazy-loaded Singletons ──────────────────────────────────────────────────

_embedding_model = None
_cross_encoder = None
_qdrant_client = None
_bm25_data = None

BM25_PATH = os.path.join("data", "bm25_index.pkl")
QDRANT_COLLECTION = "arxiv_papers"


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model: all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def get_cross_encoder() -> CrossEncoder:
    """Lazy-load the cross-encoder reranker."""
    global _cross_encoder
    if _cross_encoder is None:
        print("Loading cross-encoder: ms-marco-MiniLM-L-6-v2")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def get_qdrant_client() -> QdrantClient:
    """Lazy-load the Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        if not url or not api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
        print(f"Connecting to Qdrant at {url}")
        _qdrant_client = QdrantClient(url=url, api_key=api_key)
    return _qdrant_client


def get_bm25_data() -> dict:
    """Lazy-load the BM25 index from disk."""
    global _bm25_data
    if _bm25_data is None:
        if not os.path.exists(BM25_PATH):
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_PATH}. Run ingestion.py first."
            )
        print(f"Loading BM25 index from {BM25_PATH}")
        with open(BM25_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


# ─── 1. Dense Search ─────────────────────────────────────────────────────────

def dense_search(query: str, top_k: int = 20) -> list[dict]:
    """Search Qdrant with a dense vector query."""

    model = get_embedding_model()
    client = get_qdrant_client()

    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
    )

    output = []
    for point in results.points:
        payload = point.payload or {}
        text = payload.get("chunk_text", "")
        metadata = {k: v for k, v in payload.items() if k != "chunk_text"}
        output.append({
            "id": point.id,
            "text": text,
            "metadata": metadata,
            "score": point.score,
        })

    return output


# ─── 2. Sparse Search ────────────────────────────────────────────────────────

def sparse_search(query: str, top_k: int = 20) -> list[dict]:
    """Search using the BM25 index (sparse retrieval)."""

    bm25_data = get_bm25_data()
    bm25: BM25Okapi = bm25_data["bm25"]
    corpus: list[str] = bm25_data["corpus"]
    metadata_list: list[dict] = bm25_data["metadata"]

    # Tokenize query the same way as ingestion
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top_k indices sorted by score descending
    top_indices = np.argsort(scores)[::-1][:top_k]

    output = []
    for idx in top_indices:
        idx = int(idx)
        output.append({
            "id": idx,
            "text": corpus[idx],
            "metadata": metadata_list[idx],
            "score": float(scores[idx]),
        })

    return output


# ─── 3. Reciprocal Rank Fusion ───────────────────────────────────────────────

def rrf_merge(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge dense and sparse results using Reciprocal Rank Fusion."""

    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    # Score from dense results
    for rank, doc in enumerate(dense_results):
        key = doc["text"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in doc_map:
            doc_map[key] = doc

    # Score from sparse results
    for rank, doc in enumerate(sparse_results):
        key = doc["text"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in doc_map:
            doc_map[key] = doc

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    merged = []
    for key in sorted_keys:
        doc = doc_map[key].copy()
        doc["score"] = rrf_scores[key]
        merged.append(doc)

    return merged


# ─── 4. Cross-Encoder Reranking ──────────────────────────────────────────────

def sigmoid(x: float) -> float:
    """Apply sigmoid to normalize scores to 0-1."""
    return 1.0 / (1.0 + np.exp(-x))


def rerank(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank results using a cross-encoder model."""

    if not results:
        return []

    # Take top-10 candidates for reranking
    candidates = results[:10]
    cross_encoder = get_cross_encoder()

    # Build (query, text) pairs
    pairs = [(query, doc["text"]) for doc in candidates]
    raw_scores = cross_encoder.predict(pairs)

    # Apply sigmoid and attach rerank_score
    reranked = []
    for doc, raw_score in zip(candidates, raw_scores):
        doc_copy = doc.copy()
        doc_copy["rerank_score"] = float(sigmoid(float(raw_score)))
        reranked.append(doc_copy)

    # Sort by rerank_score descending
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_k]


# ─── 5. Hybrid Search (full pipeline) ────────────────────────────────────────

def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """Full hybrid search: dense + sparse → RRF → rerank."""

    dense = dense_search(query, 20)
    sparse = sparse_search(query, 20)
    merged = rrf_merge(dense, sparse)
    reranked = rerank(query, merged, top_k)
    return reranked


# ─── 6. Dense-Only Search ────────────────────────────────────────────────────

def dense_only_search(query: str, top_k: int = 5) -> list[dict]:
    """Dense search with reranking (no sparse/BM25)."""

    dense = dense_search(query, 20)
    reranked = rerank(query, dense, top_k)
    return reranked


# ─── 7. Hybrid No-Rerank Search ──────────────────────────────────────────────

def hybrid_no_rerank_search(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid search without cross-encoder reranking."""

    dense = dense_search(query, 20)
    sparse = sparse_search(query, 20)
    merged = rrf_merge(dense, sparse)
    return merged[:top_k]


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_query = "transformer models for natural language processing"

    print("=" * 60)
    print(f"Testing hybrid search with query:")
    print(f"  '{sample_query}'")
    print("=" * 60)

    results = hybrid_search(sample_query, top_k=5)

    print(f"\nTop {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        title = r["metadata"].get("title", "N/A")
        rerank_score = r.get("rerank_score", r.get("score", 0.0))
        print(f"  {i}. [{rerank_score:.4f}] {title}")
        print(f"     ArXiv: {r['metadata'].get('arxiv_id', 'N/A')}")
        print()

    print("=" * 60)
    print("Phase 2 complete — ready for Phase 3")
    print("=" * 60)
