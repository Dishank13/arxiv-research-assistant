"""
Phase 4: Agentic RAG with LangGraph-based CRAG routing.

This module implements the core agent logic using LangGraph's StateGraph.
The agent follows a Corrective RAG (CRAG) pattern:
  1. Retrieve documents via hybrid search
  2. Grade retrieval quality
  3. Optionally augment with web search
  4. Generate an answer
  5. Check for hallucinations
"""

import os
import sys
import time
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval import hybrid_search, dense_only_search, hybrid_no_rerank_search
from src.crag import evaluate_retrieval, get_crag_context


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    documents: list
    crag_branch: str
    web_results: list
    context: str
    answer: str
    confidence: float
    sources: list
    chat_history: list  # list of {role, content} dicts, last 5 turns
    latency_ms: float
    retrieval_mode: str  # 'full' | 'dense_only' | 'hybrid_no_rerank'


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=2048,
)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def retrieve_node(state: dict) -> dict:
    """Retrieve documents using the configured search mode."""
    mode = state.get('retrieval_mode', 'full')
    query = state['query']
    if mode == 'dense_only':
        results = dense_only_search(query)
    elif mode == 'hybrid_no_rerank':
        results = hybrid_no_rerank_search(query)
    else:
        results = hybrid_search(query)
    return {'documents': results}


def grade_node(state: dict) -> dict:
    """Grade retrieval quality using CRAG evaluation."""
    evaluation = evaluate_retrieval(state['query'], state['documents'])
    context = get_crag_context(evaluation)
    return {
        'crag_branch': evaluation.get('branch', 'CORRECT'),
        'confidence': evaluation.get('confidence_score', 0.0),
        'web_results': evaluation.get('web_results', []),
        'documents': evaluation.get('documents', state['documents']),
        'context': context,
    }


def web_search_node(state: dict) -> dict:
    """Pass-through node — web search is handled inside grade_node via crag.py."""
    return {
        'web_results': state.get('web_results', []),
    }


def generate_node(state: dict) -> dict:
    """Generate an answer using the LLM with CRAG context."""
    system_msg = SystemMessage(
        content=(
            "You are a helpful ML research assistant. Answer questions about "
            "machine learning papers accurately based on the provided context. "
            "If you cannot find the answer in the context, say so clearly. "
            "Always cite specific papers when possible."
        )
    )

    # Build history messages (last 5 turns)
    history_msgs = []
    for turn in (state.get('chat_history') or [])[-5:]:
        role = turn.get('role', 'user')
        content = turn.get('content', '')
        if role == 'user':
            history_msgs.append(HumanMessage(content=content))
        else:
            from langchain_core.messages import AIMessage
            history_msgs.append(AIMessage(content=content))

    # Include the CRAG context in the human message
    context = state.get('context', '')
    human_content = f"Context:\n{context}\n\nQuestion: {state['query']}"
    human_msg = HumanMessage(content=human_content)

    response = llm.invoke([system_msg, *history_msgs, human_msg])

    # Extract sources from documents
    sources = []
    for doc in (state.get('documents') or []):
        metadata = doc.get('metadata', doc) if isinstance(doc, dict) else {}
        sources.append({
            'arxiv_id': metadata.get('arxiv_id', ''),
            'title': metadata.get('title', ''),
            'authors': metadata.get('authors', ''),
            'published_date': metadata.get('published_date', ''),
            'pdf_url': metadata.get('pdf_url', ''),
            'relevance_score': metadata.get('relevance_score', doc.get('score', 0.0) if isinstance(doc, dict) else 0.0),
        })

    return {
        'answer': response.content,
        'sources': sources,
    }


def check_hallucination_node(state: dict) -> dict:
    """Verify the answer is grounded in the provided context."""
    context = state.get('context', '')
    answer = state.get('answer', '')

    verification_prompt = (
        f"Given this context:\n{context}\n\n"
        f"And this answer:\n{answer}\n\n"
        "Is the answer fully supported by the context? "
        "Reply with just YES or NO followed by a brief explanation."
    )

    response = llm.invoke([HumanMessage(content=verification_prompt)])
    verdict = response.content.strip()

    if verdict.upper().startswith('NO'):
        updated_answer = (
            f"{answer}\n\n"
            "⚠️ *Disclaimer: Parts of this answer may not be fully supported "
            "by the retrieved context. Please verify critical claims against "
            "the original papers.*"
        )
        return {'answer': updated_answer}

    return {'answer': answer}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_grade(state: dict) -> str:
    """Route to generate or web_search based on CRAG grading."""
    if state['crag_branch'] == 'CORRECT':
        return 'generate'
    else:
        return 'web_search'


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node('retrieve', retrieve_node)
    workflow.add_node('grade', grade_node)
    workflow.add_node('web_search', web_search_node)
    workflow.add_node('generate', generate_node)
    workflow.add_node('check_hallucination', check_hallucination_node)

    workflow.set_entry_point('retrieve')
    workflow.add_edge('retrieve', 'grade')
    workflow.add_conditional_edges(
        'grade',
        route_after_grade,
        {'generate': 'generate', 'web_search': 'web_search'},
    )
    workflow.add_edge('web_search', 'generate')
    workflow.add_edge('generate', 'check_hallucination')
    workflow.add_edge('check_hallucination', END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Main query function
# ---------------------------------------------------------------------------

def query_agent(
    query: str,
    chat_history: list = None,
    retrieval_mode: str = 'full',
) -> dict:
    """
    Main entry point for querying the agent.

    Args:
        query: The user's question.
        chat_history: List of {role, content} dicts for multi-turn context.
        retrieval_mode: 'full' | 'dense_only' | 'hybrid_no_rerank'

    Returns:
        dict with answer, sources, crag_branch, confidence, latency_ms, etc.
    """
    if chat_history is None:
        chat_history = []

    # Keep only last 5 turns
    chat_history = chat_history[-5:]

    start_time = time.time()

    graph = build_graph()

    initial_state = {
        'query': query,
        'documents': [],
        'crag_branch': '',
        'web_results': [],
        'context': '',
        'answer': '',
        'confidence': 0.0,
        'sources': [],
        'chat_history': chat_history,
        'latency_ms': 0.0,
        'retrieval_mode': retrieval_mode,
    }

    result = graph.invoke(initial_state)

    latency_ms = (time.time() - start_time) * 1000
    result['latency_ms'] = latency_ms

    return result


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing Agentic RAG agent...")
    print("=" * 60)

    test_query = "What is the transformer architecture?"
    result = query_agent(test_query)

    print(f"\nQuery: {test_query}")
    print(f"\nAnswer:\n{result.get('answer', 'N/A')}")
    print(f"\nSources ({len(result.get('sources', []))}):")
    for i, src in enumerate(result.get('sources', []), 1):
        print(f"  {i}. {src.get('title', 'Unknown')} (arxiv: {src.get('arxiv_id', 'N/A')})")
    print(f"\nCRAG Branch: {result.get('crag_branch', 'N/A')}")
    print(f"Confidence:  {result.get('confidence', 0.0):.2f}")
    print(f"Latency:     {result.get('latency_ms', 0.0):.0f} ms")
    print("=" * 60)
    print("Phase 4 complete — ready for Phase 5")
