"""
Phase 1: Data Ingestion Pipeline
=================================
Fetches ML/AI papers from ArXiv, creates text chunks,
generates embeddings, uploads to Qdrant Cloud, and builds a BM25 index.
"""

import os
import json
import time
import pickle
import urllib.request
import urllib.parse
import numpy as np
import feedparser
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct
from rank_bm25 import BM25Okapi

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
ARXIV_QUERY = "cat:cs.AI OR cat:cs.CL OR cat:cs.LG"
DATE_FILTER = "submittedDate:[202001010000 TO 202412312359]"
BATCH_SIZE = 100
TARGET_PAPERS = 3000
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 4, 8]

RAW_DIR = os.path.join("data", "raw")
CHUNKS_DIR = os.path.join("data", "chunks")
RAW_FILE = os.path.join(RAW_DIR, "papers.jsonl")
CHUNKS_FILE = os.path.join(CHUNKS_DIR, "chunks.jsonl")
BM25_FILE = os.path.join("data", "bm25_index.pkl")

QDRANT_COLLECTION = "arxiv_papers"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


# ─── Step 1: Fetch Papers from ArXiv API ──────────────────────────────────────

def fetch_papers_from_arxiv(target: int = TARGET_PAPERS) -> list[dict]:
    """Fetch papers from the ArXiv API with pagination, retry logic, and dedup."""

    os.makedirs(RAW_DIR, exist_ok=True)

    all_papers = []
    seen_ids = set()  # for deduplication (version-stripped)
    start = 0

    print(f"Starting ArXiv fetch — target: {target} papers")

    while len(all_papers) < target:
        search_query = f"({ARXIV_QUERY}) AND {DATE_FILTER}"
        params = urllib.parse.urlencode({
            "search_query": search_query,
            "start": start,
            "max_results": BATCH_SIZE,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        url = f"{ARXIV_BASE_URL}?{params}"

        # Retry logic with exponential backoff
        response_text = None
        for attempt in range(MAX_RETRIES):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    response_text = resp.read().decode("utf-8")
                break
            except Exception as e:
                wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 8
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} after error: {e} — waiting {wait}s")
                time.sleep(wait)

        if response_text is None:
            print("Failed to fetch after retries. Stopping.")
            break

        feed = feedparser.parse(response_text)
        entries = feed.entries

        if not entries:
            print("No more entries returned by API. Stopping.")
            break

        batch_count = 0
        for entry in entries:
            # Extract arxiv_id — strip URL prefix
            raw_id = entry.id  # e.g. 'http://arxiv.org/abs/2301.12345v1'
            arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

            # Dedup key: strip version suffix (e.g. 'v1', 'v2')
            base_id = arxiv_id.rsplit("v", 1)[0] if "v" in arxiv_id else arxiv_id
            if base_id in seen_ids:
                continue
            seen_ids.add(base_id)

            # Authors
            authors = [a.get("name", "") for a in entry.get("authors", [])]

            # Abstract — clean whitespace
            abstract = entry.get("summary", "").strip()
            abstract = " ".join(abstract.split())

            # Categories
            primary_cat = entry.get("arxiv_primary_category", {}).get("term", "")
            tags = [t.get("term", "") for t in entry.get("tags", [])]
            categories = list(set([primary_cat] + tags)) if primary_cat else tags

            # Published date
            published_date = entry.get("published", "")

            # PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            paper = {
                "arxiv_id": arxiv_id,
                "title": entry.get("title", "").strip().replace("\n", " "),
                "authors": authors,
                "abstract": abstract,
                "categories": categories,
                "published_date": published_date,
                "pdf_url": pdf_url,
            }
            all_papers.append(paper)
            batch_count += 1

            if len(all_papers) >= target:
                break

        start += BATCH_SIZE

        # Progress update
        if len(all_papers) % 100 == 0 or len(all_papers) >= target:
            print(f"Fetched {len(all_papers)}/{target} papers...")

        # Rate limiting
        time.sleep(3)

    # Save to JSONL
    with open(RAW_FILE, "w", encoding="utf-8") as f:
        for paper in all_papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_papers)} papers to {RAW_FILE}")
    return all_papers


# ─── Step 2: Create Chunks ───────────────────────────────────────────────────

def create_chunks(papers: list[dict]) -> tuple[list[str], list[dict]]:
    """Create text chunks from papers and save to JSONL."""

    os.makedirs(CHUNKS_DIR, exist_ok=True)

    chunk_texts = []
    chunk_metadata = []

    for paper in papers:
        text = (
            f"Title: {paper['title']}\n"
            f"Authors: {', '.join(paper['authors'])}\n"
            f"Abstract: {paper['abstract']}"
        )
        metadata = {
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "published_date": paper["published_date"],
            "categories": paper["categories"],
            "pdf_url": paper["pdf_url"],
            "authors": paper["authors"],
        }
        chunk_texts.append(text)
        chunk_metadata.append(metadata)

    # Save chunks to JSONL
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for text, meta in zip(chunk_texts, chunk_metadata):
            record = {"chunk_text": text, "metadata": meta}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Created {len(chunk_texts)} chunks -> {CHUNKS_FILE}")
    return chunk_texts, chunk_metadata


# ─── Step 3: Generate Embeddings ─────────────────────────────────────────────

def generate_embeddings(chunk_texts: list[str]) -> np.ndarray:
    """Generate embeddings using SentenceTransformers."""

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Encoding {len(chunk_texts)} chunks...")
    embeddings = model.encode(chunk_texts, batch_size=64, show_progress_bar=True)
    embeddings = np.array(embeddings)

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


# ─── Step 4: Upload to Qdrant Cloud ──────────────────────────────────────────

def upload_to_qdrant(
    embeddings: np.ndarray,
    chunk_texts: list[str],
    chunk_metadata: list[dict],
    papers: list[dict],
) -> None:
    """Upload vectors and metadata to Qdrant Cloud."""

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")

    print(f"Connecting to Qdrant at {qdrant_url}")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Delete existing collection if it exists
    try:
        client.delete_collection(QDRANT_COLLECTION)
        print(f"Deleted existing collection '{QDRANT_COLLECTION}'")
    except Exception:
        pass

    # Create collection
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection '{QDRANT_COLLECTION}' (dim={EMBEDDING_DIM}, cosine)")

    # Build a lookup from papers for abstract
    paper_lookup = {p["arxiv_id"]: p for p in papers}

    # Upsert in batches of 100
    batch_size = 100
    total = len(embeddings)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        points = []

        for i in range(batch_start, batch_end):
            meta = chunk_metadata[i]
            paper = paper_lookup.get(meta["arxiv_id"], {})

            payload = {
                "arxiv_id": meta["arxiv_id"],
                "title": meta["title"],
                "authors": meta["authors"],
                "abstract": paper.get("abstract", ""),
                "categories": meta["categories"],
                "published_date": meta["published_date"],
                "pdf_url": meta["pdf_url"],
                "chunk_text": chunk_texts[i],
            }

            points.append(
                PointStruct(
                    id=i,
                    vector=embeddings[i].tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"  Upserted batch {batch_start}–{batch_end - 1} ({batch_end}/{total})")

    print(f"Uploaded {total} vectors to Qdrant collection '{QDRANT_COLLECTION}'")


# ─── Step 5: Build BM25 Index ────────────────────────────────────────────────

def build_bm25_index(
    chunk_texts: list[str], chunk_metadata: list[dict]
) -> None:
    """Build and save a BM25 index from chunk texts."""

    print("Building BM25 index...")

    # Tokenize: lowercase and split on whitespace
    tokenized_corpus = [text.lower().split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # Save as pickle
    bm25_data = {
        "bm25": bm25,
        "corpus": chunk_texts,
        "metadata": chunk_metadata,
    }

    os.makedirs(os.path.dirname(BM25_FILE), exist_ok=True)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"BM25 index saved to {BM25_FILE} ({len(chunk_texts)} documents)")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    overall_start = time.time()

    # Step 1: Fetch papers
    print("=" * 60)
    print("STEP 1: Fetching papers from ArXiv")
    print("=" * 60)
    t0 = time.time()
    papers = fetch_papers_from_arxiv()
    print(f"  [TIME] Step 1 took {time.time() - t0:.1f}s\n")

    # Step 2: Create chunks
    print("=" * 60)
    print("STEP 2: Creating text chunks")
    print("=" * 60)
    t0 = time.time()
    chunk_texts, chunk_metadata = create_chunks(papers)
    print(f"  [TIME] Step 2 took {time.time() - t0:.1f}s\n")

    # Step 3: Generate embeddings
    print("=" * 60)
    print("STEP 3: Generating embeddings")
    print("=" * 60)
    t0 = time.time()
    embeddings = generate_embeddings(chunk_texts)
    print(f"  [TIME] Step 3 took {time.time() - t0:.1f}s\n")

    # Step 4: Upload to Qdrant
    print("=" * 60)
    print("STEP 4: Uploading to Qdrant Cloud")
    print("=" * 60)
    t0 = time.time()
    upload_to_qdrant(embeddings, chunk_texts, chunk_metadata, papers)
    print(f"  [TIME] Step 4 took {time.time() - t0:.1f}s\n")

    # Step 5: Build BM25 index
    print("=" * 60)
    print("STEP 5: Building BM25 index")
    print("=" * 60)
    t0 = time.time()
    build_bm25_index(chunk_texts, chunk_metadata)
    print(f"  [TIME] Step 5 took {time.time() - t0:.1f}s\n")

    total_time = time.time() - overall_start
    print("=" * 60)
    print(f"Phase 1 complete — ready for Phase 2")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 60)
