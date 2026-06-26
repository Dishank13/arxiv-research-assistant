"""
FastAPI backend for the ArXiv RAG Research Assistant.

Includes rate limiting, input validation, CORS hardening, and request logging.

Endpoints:
  POST /query               -- Ask a question, get an answer with sources
  GET  /session/{id}/history -- Retrieve session conversation history
  POST /evaluate             -- Run the evaluation pipeline
  GET  /health               -- Health check
"""

import os
import sys
import time
import hashlib
import logging
import re
from datetime import datetime
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import settings

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("rag_api")

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    USE_SLOWAPI = True
    logger.info("slowapi rate limiter loaded")
except ImportError:
    USE_SLOWAPI = False
    logger.warning("slowapi not installed -- using simple in-memory rate limiter")

    _request_timestamps: dict[str, list[float]] = defaultdict(list)

    def simple_rate_check(request: Request, limit: int = 20) -> None:
        """Simple per-IP rate limiter fallback."""
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        # Keep only timestamps from the last 60 seconds
        _request_timestamps[ip] = [
            t for t in _request_timestamps[ip] if now - t < 60
        ]
        if len(_request_timestamps[ip]) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait.",
            )
        _request_timestamps[ip].append(now)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ArXiv Research Assistant API",
    version="1.0.0",
    description="CRAG-powered research assistant for ML papers.",
)

if USE_SLOWAPI:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS -- restricted origins (not wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "https://Dishank13-arxiv-research-assistant.hf.space",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session storage (in-memory)
# ---------------------------------------------------------------------------
sessions: dict[str, list[dict]] = defaultdict(list)

# ---------------------------------------------------------------------------
# Pydantic models with validation
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        # Strip whitespace and control characters
        v = v.strip()
        v = "".join(c for c in v if c.isprintable())
        if not v:
            raise ValueError("Query cannot be empty.")
        if len(v) > settings.max_query_length:
            raise ValueError(
                f"Query too long. Maximum {settings.max_query_length} characters."
            )
        return v

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            return "default"
        if len(v) > 100:
            raise ValueError("Session ID too long.")
        # Only allow alphanumeric, hyphens, underscores
        v = re.sub(r"[^a-zA-Z0-9_\-]", "", v)
        return v or "default"


class QueryResponse(BaseModel):
    answer: str
    sources: list
    crag_branch: str
    confidence: float
    latency_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_session(session_id: str) -> str:
    """Hash session ID for safe logging (no PII)."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Global exception handler -- never expose stack traces
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong. Please try again."},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: Request, body: QueryRequest):
    """Process a user query through the CRAG agent."""
    # Rate limit check
    if not USE_SLOWAPI:
        simple_rate_check(request, settings.rate_limit_per_minute)

    try:
        from src.agent import query_agent

        # Build chat_history from session for multi-turn context
        chat_history = []
        for entry in sessions[body.session_id]:
            chat_history.append({"role": "user", "content": entry["query"]})
            chat_history.append({"role": "assistant", "content": entry["answer"]})

        start_time = time.time()
        result = query_agent(query=body.query, chat_history=chat_history)
        latency_ms = (time.time() - start_time) * 1000

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        crag_branch = result.get("crag_branch", "")
        confidence = result.get("confidence", 0.0)

        # Store the interaction in session history
        sessions[body.session_id].append({
            "query": body.query,
            "answer": answer,
            "sources": sources,
            "crag_branch": crag_branch,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
        })

        # Log request metadata (no query text for privacy)
        logger.info(
            f"query | session={_hash_session(body.session_id)} "
            f"| len={len(body.query)} | branch={crag_branch} "
            f"| confidence={confidence:.4f} | latency={latency_ms:.0f}ms"
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            crag_branch=crag_branch,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again.",
        )


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Retrieve the conversation history for a given session."""
    return {
        "session_id": session_id,
        "history": sessions.get(session_id, []),
        "total_queries": len(sessions.get(session_id, [])),
    }


@app.post("/evaluate")
async def run_evaluation_endpoint():
    """Run the evaluation pipeline and return results."""
    try:
        from src.evaluation import run_evaluation
        results = run_evaluation()
        return results

    except ImportError as e:
        logger.error(f"Evaluation module error: {e}")
        raise HTTPException(
            status_code=501,
            detail="Evaluation module not available.",
        )
    except Exception as e:
        logger.error(f"Evaluation error: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Evaluation failed. Please try again.",
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting ArXiv Research Assistant API on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
