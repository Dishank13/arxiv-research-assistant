---
title: ArXiv Research Assistant
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---
# ArXiv Research Assistant

An agentic RAG system for querying ML/AI research papers. Combines hybrid retrieval (dense + sparse), corrective grading with web search fallback, cross-encoder reranking, and a LangGraph-powered agent loop with hallucination detection.

## Architecture

Queries flow through a multi-stage pipeline: hybrid retrieval merges Qdrant dense search with BM25 sparse search via Reciprocal Rank Fusion, then a cross-encoder reranks the top candidates. A CRAG (Corrective RAG) module scores retrieval confidence and routes to Tavily web search when local context is insufficient. The LangGraph agent generates answers with Llama 3.3 70B via Groq and performs a hallucination self-check before responding.

## Quick Start

### Prerequisites
- Python 3.10+
- API keys for Groq, Qdrant Cloud, Tavily, and HuggingFace

### Setup

1. Clone the repository
```bash
git clone <repo-url>
cd RAG-ML-Papers
```

2. Create environment file
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run data ingestion
```bash
python src/ingestion.py
```
This fetches ~3,000 ArXiv papers, generates embeddings, uploads to Qdrant, and builds the BM25 index. Takes approximately 15-20 minutes.

5. Start the API server
```bash
uvicorn api.main:app --port 8000
```

6. Launch the frontend
```bash
streamlit run frontend/app.py
```

The app opens at `http://localhost:8501`.

## Environment Variables

Create a `.env` file (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key for LLM inference |
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `TAVILY_API_KEY` | Tavily API key for web search |
| `HF_TOKEN` | HuggingFace token for model downloads |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Submit a research question |
| GET | `/session/{id}/history` | Retrieve session history |
| POST | `/evaluate` | Run RAGAS evaluation |
| GET | `/health` | Health check |

## Evaluation

Run the evaluation suite with RAGAS metrics:

```bash
# Full evaluation (25 questions, 3 ablation studies)
python src/evaluation.py

# Quick evaluation (5 questions)
python src/evaluation.py --quick 5
```

Ablation configurations:
- **Full Pipeline**: Dense + BM25 + RRF + Cross-encoder reranking
- **Dense Only**: Qdrant search + Cross-encoder reranking
- **Hybrid No Rerank**: Dense + BM25 + RRF, no cross-encoder

## Tech Stack

| Component | Technology |
|-----------|------------|
| Vector Database | Qdrant Cloud |
| Embeddings | all-MiniLM-L6-v2 (384d) |
| LLM | Llama 3.3 70B via Groq |
| Reranker | ms-marco-MiniLM cross-encoder |
| Agent Framework | LangGraph |
| Web Search | Tavily API |
| Evaluation | RAGAS |
| Backend | FastAPI |
| Frontend | Streamlit |

## License

MIT License
