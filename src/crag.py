"""
Phase 3: CRAG (Corrective Retrieval-Augmented Generation)
==========================================================
Evaluates retrieval quality using confidence scoring and branches
to web search when retrieval is ambiguous or incorrect.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

CRAG_LOG_FILE = os.path.join("data", "crag_log.jsonl")

CORRECT_THRESHOLD = 0.7
AMBIGUOUS_THRESHOLD = 0.3


# ─── 1. Evaluate Retrieval Quality ───────────────────────────────────────────

def evaluate_retrieval(query: str, results: list[dict]) -> dict:
    """
    Evaluate retrieval quality and apply CRAG branching logic.

    - CORRECT (score > 0.7): Trust retrieved documents.
    - AMBIGUOUS (0.3 <= score <= 0.7): Supplement with web search.
    - INCORRECT (score < 0.3): Fall back to web search only.
    """

    # Compute confidence score from rerank scores
    if results:
        confidence_score = max(
            r.get("rerank_score", 0.0) for r in results
        )
    else:
        confidence_score = 0.0

    # Determine branch
    if confidence_score > CORRECT_THRESHOLD:
        branch = "CORRECT"
    elif confidence_score >= AMBIGUOUS_THRESHOLD:
        branch = "AMBIGUOUS"
    else:
        branch = "INCORRECT"

    print(f"CRAG evaluation: confidence={confidence_score:.4f} → branch={branch}")

    # Web search for AMBIGUOUS and INCORRECT branches
    web_results = []
    if branch in ("AMBIGUOUS", "INCORRECT"):
        print(f"  Triggering web search for query: '{query}'")
        web_results = web_search(query)

    # Determine which documents to keep
    if branch == "CORRECT":
        documents = results
    elif branch == "AMBIGUOUS":
        documents = results
    else:  # INCORRECT
        documents = []

    # Log to CRAG log file
    _log_crag_decision(query, branch, confidence_score)

    return {
        "branch": branch,
        "confidence_score": confidence_score,
        "web_results": web_results,
        "documents": documents,
    }


# ─── 2. Web Search ───────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 3) -> list[str]:
    """
    Search the web using Tavily API for supplementary information.
    Returns a list of content strings.
    """

    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("  WARNING: TAVILY_API_KEY not set in .env — skipping web search")
            return []

        tavily = TavilyClient(api_key=api_key)
        response = tavily.search(query=query, max_results=max_results)

        contents = []
        for result in response.get("results", []):
            content = result.get("content", "")
            if content:
                contents.append(content)

        print(f"  Web search returned {len(contents)} results")
        return contents

    except Exception as e:
        print(f"  WARNING: Web search failed: {e}")
        return []


# ─── 3. Build CRAG Context ───────────────────────────────────────────────────

def get_crag_context(crag_result: dict) -> str:
    """
    Build a context string based on the CRAG branch.

    - CORRECT: Only retrieved documents.
    - AMBIGUOUS: Retrieved documents + web results.
    - INCORRECT: Only web results.
    """

    branch = crag_result["branch"]
    documents = crag_result.get("documents", [])
    web_results = crag_result.get("web_results", [])

    context_parts = []

    if branch == "CORRECT":
        # Only retrieved documents
        context_parts.append("=== Retrieved Documents ===\n")
        for i, doc in enumerate(documents, 1):
            title = doc.get("metadata", {}).get("title", "Untitled")
            text = doc.get("text", "")
            score = doc.get("rerank_score", doc.get("score", 0.0))
            context_parts.append(
                f"[Document {i}] (score: {score:.4f})\n"
                f"Title: {title}\n"
                f"{text}\n"
            )

    elif branch == "AMBIGUOUS":
        # Retrieved documents + web results
        context_parts.append("=== Retrieved Documents ===\n")
        for i, doc in enumerate(documents, 1):
            title = doc.get("metadata", {}).get("title", "Untitled")
            text = doc.get("text", "")
            score = doc.get("rerank_score", doc.get("score", 0.0))
            context_parts.append(
                f"[Document {i}] (score: {score:.4f})\n"
                f"Title: {title}\n"
                f"{text}\n"
            )

        if web_results:
            context_parts.append("\n=== Web Search Results ===\n")
            for i, content in enumerate(web_results, 1):
                context_parts.append(f"[Web Result {i}]\n{content}\n")

    elif branch == "INCORRECT":
        # Only web results
        if web_results:
            context_parts.append("=== Web Search Results ===\n")
            for i, content in enumerate(web_results, 1):
                context_parts.append(f"[Web Result {i}]\n{content}\n")
        else:
            context_parts.append(
                "No relevant documents or web results found for this query.\n"
            )

    return "\n".join(context_parts)


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _log_crag_decision(query: str, branch: str, confidence_score: float) -> None:
    """Append a CRAG decision to the log file."""

    os.makedirs(os.path.dirname(CRAG_LOG_FILE), exist_ok=True)

    log_entry = {
        "query": query,
        "branch": branch,
        "confidence_score": confidence_score,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        with open(CRAG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  WARNING: Failed to write CRAG log: {e}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Testing CRAG module with mock results")
    print("=" * 60)

    # Mock retrieval results to test each branch
    mock_results_correct = [
        {
            "id": 0,
            "text": "Title: Attention Is All You Need\nAuthors: Vaswani et al.\nAbstract: We propose a new architecture...",
            "metadata": {"title": "Attention Is All You Need", "arxiv_id": "1706.03762"},
            "score": 0.95,
            "rerank_score": 0.85,
        },
        {
            "id": 1,
            "text": "Title: BERT: Pre-training of Deep Bidirectional Transformers\nAuthors: Devlin et al.\nAbstract: We introduce BERT...",
            "metadata": {"title": "BERT", "arxiv_id": "1810.04805"},
            "score": 0.90,
            "rerank_score": 0.78,
        },
    ]

    mock_results_ambiguous = [
        {
            "id": 2,
            "text": "Title: Some Tangentially Related Paper\nAuthors: Smith et al.\nAbstract: This paper discusses...",
            "metadata": {"title": "Some Paper", "arxiv_id": "2301.00001"},
            "score": 0.55,
            "rerank_score": 0.50,
        },
    ]

    mock_results_incorrect = [
        {
            "id": 3,
            "text": "Title: Completely Irrelevant Paper\nAuthors: Doe et al.\nAbstract: Unrelated topic...",
            "metadata": {"title": "Irrelevant Paper", "arxiv_id": "2301.99999"},
            "score": 0.10,
            "rerank_score": 0.15,
        },
    ]

    # Test CORRECT branch
    print("\n--- Test: CORRECT branch ---")
    result = evaluate_retrieval("transformer architectures", mock_results_correct)
    context = get_crag_context(result)
    print(f"Branch: {result['branch']}, Confidence: {result['confidence_score']:.4f}")
    print(f"Context preview:\n{context[:200]}...\n")

    # Test AMBIGUOUS branch
    print("--- Test: AMBIGUOUS branch ---")
    result = evaluate_retrieval("novel optimization techniques", mock_results_ambiguous)
    context = get_crag_context(result)
    print(f"Branch: {result['branch']}, Confidence: {result['confidence_score']:.4f}")
    print(f"Web results count: {len(result['web_results'])}\n")

    # Test INCORRECT branch
    print("--- Test: INCORRECT branch ---")
    result = evaluate_retrieval("quantum computing breakthroughs 2024", mock_results_incorrect)
    context = get_crag_context(result)
    print(f"Branch: {result['branch']}, Confidence: {result['confidence_score']:.4f}")
    print(f"Web results count: {len(result['web_results'])}\n")

    print("=" * 60)
    print("Phase 3 complete — ready for Phase 4")
    print("=" * 60)
