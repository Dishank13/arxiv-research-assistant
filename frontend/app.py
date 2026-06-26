import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import hashlib

from src.agent import query_agent

# ---------------------------------------------------------------------------
# Global Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATION_RESULTS_PATH = os.path.join(PROJECT_ROOT, "eval", "ablation_results.json")
TEST_QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "eval", "test_questions.json")

st.set_page_config(
    page_title="ArXiv Research Assistant",
    page_icon="\U0001f52c",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ---- Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
    color: #e2e8f0;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stMarkdown {
    color: #e2e8f0;
}

/* ---- Main Header ---- */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: #ffffff;
}
.main-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    color: #ffffff;
}
.main-header p {
    font-size: 0.95rem;
    margin: 0;
    opacity: 0.9;
    color: #f0f0ff;
}

/* ---- CRAG Badges ---- */
.badge-correct {
    display: inline-block;
    background: #10b981;
    color: #fff;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-ambiguous {
    display: inline-block;
    background: #f59e0b;
    color: #1a1a2e;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-incorrect {
    display: inline-block;
    background: #ef4444;
    color: #fff;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}

/* ---- Source Cards ---- */
.source-card {
    background: linear-gradient(135deg, #1e1e3f 0%, #2a2a4a 100%);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.source-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15);
}
.source-card h4 {
    margin: 0 0 0.5rem 0;
    color: #e2e8f0;
    font-size: 1rem;
}
.source-card p {
    margin: 0.2rem 0;
    color: #a0aec0;
    font-size: 0.85rem;
}
.source-card a {
    color: #818cf8;
    text-decoration: none;
    font-weight: 500;
}
.source-card a:hover {
    text-decoration: underline;
}

/* ---- Metric Cards ---- */
.metric-card {
    background: linear-gradient(135deg, #1e1e3f 0%, #2a2a4a 100%);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 14px;
    padding: 1.5rem 1rem;
    text-align: center;
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.3rem 0;
}
.metric-card .metric-label {
    font-size: 0.85rem;
    color: #a0aec0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ---- Chat Bubbles ---- */
.chat-user {
    border-left: 4px solid #667eea;
    padding: 0.8rem 1.2rem;
    margin: 0.6rem 0;
    background: rgba(102, 126, 234, 0.06);
    border-radius: 0 10px 10px 0;
}
.chat-assistant {
    border-left: 4px solid #764ba2;
    padding: 0.8rem 1.2rem;
    margin: 0.6rem 0;
    background: rgba(118, 75, 162, 0.06);
    border-radius: 0 10px 10px 0;
}

/* ---- Feature Cards (About page) ---- */
.feature-card {
    background: linear-gradient(135deg, #1e1e3f 0%, #2a2a4a 100%);
    border: 1px solid rgba(102, 126, 234, 0.25);
    border-radius: 14px;
    padding: 1.5rem 1.5rem;
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.feature-card:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.12);
}
.feature-card h4 {
    margin: 0 0 0.6rem 0;
    color: #e2e8f0;
    font-size: 1.05rem;
}
.feature-card p {
    margin: 0;
    color: #a0aec0;
    font-size: 0.9rem;
    line-height: 1.55;
}

/* ---- Footer ---- */
.footer-text {
    text-align: center;
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 1rem;
}

/* ---- Example Question Chips ---- */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.question-chip {
    display: inline-block;
    border: 1px solid #764ba2;
    border-radius: 999px;
    padding: 0.4rem 1rem;
    font-size: 0.85rem;
    color: #c4b5fd;
    cursor: pointer;
    transition: background 0.2s ease;
}
.question-chip:hover {
    background: rgba(118, 75, 162, 0.25);
}

/* ---- Architecture Box ---- */
.architecture-box {
    background: #0f0f23;
    border-radius: 12px;
    padding: 1.5rem;
    font-family: 'Courier New', monospace;
    color: #a78bfa;
    font-size: 0.85rem;
    line-height: 1.6;
    border: 1px solid rgba(167, 139, 250, 0.2);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Go to",
        ["Chat", "Evaluation Dashboard", "About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<small style="color:#6b7280;">ArXiv Research Assistant<br>Powered by LangGraph + Groq</small>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _crag_badge(decision: str) -> str:
    """Return an HTML span for a CRAG decision badge."""
    label = decision.strip().lower()
    if label == "correct":
        return '<span class="badge-correct">CORRECT</span>'
    elif label == "ambiguous":
        return '<span class="badge-ambiguous">AMBIGUOUS</span>'
    elif label == "incorrect":
        return '<span class="badge-incorrect">INCORRECT</span>'
    return f'<span class="badge-ambiguous">{decision.upper()}</span>'


def _build_arxiv_link(source: dict) -> str:
    """Build an ArXiv URL from a source dict."""
    pdf_url = source.get("pdf_url", "")
    arxiv_id = source.get("arxiv_id", "")
    if pdf_url:
        return pdf_url
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return ""


def _render_source_card(source: dict, idx: int) -> str:
    """Render a source card as an HTML block."""
    title = source.get("title", "Untitled")
    authors = source.get("authors", "")
    if isinstance(authors, list):
        authors = ", ".join(authors)
    published = source.get("published_date", source.get("published", ""))
    link = _build_arxiv_link(source)
    relevance = source.get("relevance_score", source.get("score", 0))

    link_html = ""
    if link:
        link_html = f'<p><a href="{link}" target="_blank">View on ArXiv &rarr;</a></p>'

    return f"""
    <div class="source-card">
        <h4>{idx}. {title}</h4>
        <p><strong>Authors:</strong> {authors}</p>
        <p><strong>Published:</strong> {published}</p>
        {link_html}
    </div>
    """


# ---------------------------------------------------------------------------
# PAGE 1: Chat
# ---------------------------------------------------------------------------
def page_chat():
    st.markdown(
        """
    <div class="main-header">
        <h1>Research Chat</h1>
        <p>Ask questions about ML/AI research papers</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Session state init
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.md5(
            str(time.time()).encode()
        ).hexdigest()[:12]

    # Example questions when chat is empty
    example_questions = [
        "What is the main idea behind the Transformer architecture?",
        "How does LoRA reduce fine-tuning memory requirements?",
        "What improvements did Mistral 7B make over LLaMA 2?",
        "Explain the key contribution of the RLHF paper by Ouyang et al.",
        "What is RAG and how does it differ from fine-tuning?",
    ]

    if not st.session_state.chat_history:
        st.markdown("#### Try asking:")
        for i, q in enumerate(example_questions):
            if st.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state["prefill_query"] = q
                st.session_state["auto_submit"] = True
                st.rerun()

    # Text input with prefill support
    default_query = st.session_state.pop("prefill_query", "")
    auto_submit = st.session_state.pop("auto_submit", False)
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Your question",
            value=default_query,
            placeholder="e.g. How does LoRA reduce fine-tuning costs?",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.button("Ask", use_container_width=True)

    # Trigger on button press OR auto-submit from chip click
    if (submitted or auto_submit) and query.strip():
        with st.spinner("Searching papers and generating answer..."):
            start_time = time.time()
            try:
                # Build chat history format for agent
                agent_history = []
                for entry in st.session_state.chat_history:
                    agent_history.append({"role": "user", "content": entry["query"]})
                    agent_history.append({"role": "assistant", "content": entry["answer"]})

                data = query_agent(query=query.strip(), chat_history=agent_history)
                elapsed = time.time() - start_time

                answer = data.get("answer", "No answer returned.")
                decision = data.get("crag_branch", "")
                confidence = data.get("confidence", None)
                sources = data.get("sources", [])

                # Display answer
                st.markdown("### Answer")
                st.markdown(answer)

                # CRAG badge + metadata row
                meta_parts = []
                if decision:
                    meta_parts.append(_crag_badge(decision))
                if confidence is not None:
                    conf_pct = (
                        f"{confidence:.0%}"
                        if isinstance(confidence, float) and confidence <= 1
                        else str(confidence)
                    )
                    meta_parts.append(
                        f'<span style="color:#a0aec0; margin-left:0.8rem;">Confidence: <strong>{conf_pct}</strong></span>'
                    )
                meta_parts.append(
                    f'<span style="color:#a0aec0; margin-left:0.8rem;">Latency: <strong>{elapsed:.1f}s</strong></span>'
                )
                st.markdown(" ".join(meta_parts), unsafe_allow_html=True)

                # Source cards
                if sources:
                    with st.expander(
                        f"Sources ({len(sources)} papers)", expanded=False
                    ):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(
                                _render_source_card(src, idx),
                                unsafe_allow_html=True,
                            )
                            relevance = src.get(
                                "relevance_score", src.get("score", 0)
                            )
                            if relevance:
                                rel_val = (
                                    float(relevance)
                                    if float(relevance) <= 1
                                    else float(relevance) / 100
                                )
                                st.progress(
                                    min(rel_val, 1.0),
                                    text=f"Relevance: {rel_val:.0%}",
                                )

                # Save to chat history
                st.session_state.chat_history.append(
                    {
                        "query": query.strip(),
                        "answer": answer,
                        "decision": decision,
                        "confidence": confidence,
                        "sources": sources,
                        "latency": elapsed,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    }
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Unable to connect to the search service. "
                    "Please ensure the backend is running."
            except Exception as e:
                st.error(f"Something went wrong. Please try again. ({e})")

    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Conversation History")
        for entry in reversed(st.session_state.chat_history):
            ts = entry.get("timestamp", "")
            st.markdown(
                f'<div class="chat-user"><strong>You</strong> '
                f'<span style="color:#6b7280; font-size:0.75rem;">{ts}</span>'
                f"<br>{entry['query']}</div>",
                unsafe_allow_html=True,
            )
            badge_html = _crag_badge(entry["decision"]) if entry.get("decision") else ""
            st.markdown(
                f'<div class="chat-assistant"><strong>Assistant</strong> {badge_html}'
                f"<br>{entry['answer']}</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# PAGE 2: Evaluation Dashboard
# ---------------------------------------------------------------------------
def page_evaluation():
    st.markdown(
        """
    <div class="main-header">
        <h1>Evaluation Dashboard</h1>
        <p>RAGAS metrics and ablation study results</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Run evaluation button
    if st.button("Run RAGAS Evaluation", use_container_width=True):
        with st.spinner("Running evaluation across test questions... this may take several minutes."):
            try:
                resp = requests.post(f"{API_URL}/evaluate", timeout=600)
                resp.raise_for_status()
                st.success("Evaluation complete! Results are displayed below.")
                st.rerun()
            except requests.exceptions.ConnectionError:
                st.error(
                    "Unable to connect to the backend. "
                    "Please ensure the API server is running."
                )
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

    # Check for existing results
    results_exist = os.path.exists(ABLATION_RESULTS_PATH)
    ablation_data = None
    if results_exist:
        try:
            with open(ABLATION_RESULTS_PATH, "r", encoding="utf-8") as f:
                ablation_data = json.load(f)
            if not ablation_data:
                results_exist = False
        except (json.JSONDecodeError, IOError):
            results_exist = False

    if not results_exist:
        st.info(
            "Evaluation hasn't been run yet. Click 'Run RAGAS Evaluation' to "
            "benchmark the pipeline across 25 test questions."
        )

        # Show preview of test questions
        if os.path.exists(TEST_QUESTIONS_PATH):
            try:
                with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
                    questions = json.load(f)
                st.markdown("#### Sample Evaluation Questions")
                q_df = pd.DataFrame(
                    [
                        {"#": i + 1, "Question": q["question"]}
                        for i, q in enumerate(questions)
                    ]
                )
                st.dataframe(q_df, use_container_width=True, hide_index=True)
            except Exception:
                pass
        return

    # ----- Display results -----
    # Determine configs — skip non-dict keys like 'timestamp'
    configs = [
        k for k, v in ablation_data.items()
        if isinstance(v, dict) and k != "timestamp"
    ]

    if not configs:
        st.warning("Ablation results file is empty or has an unexpected format.")
        return

    # Metric names to display
    METRIC_KEYS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    METRIC_LABELS = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }

    # Summary cards for the first config
    primary_key = configs[0]
    primary_metrics = ablation_data[primary_key]
    if isinstance(primary_metrics, list):
        # Average across questions
        avg = {}
        for mk in METRIC_KEYS:
            vals = [
                float(q.get(mk, 0))
                for q in primary_metrics
                if isinstance(q, dict) and mk in q
            ]
            avg[mk] = sum(vals) / len(vals) if vals else 0.0
        primary_metrics = avg
    elif not isinstance(primary_metrics, dict):
        primary_metrics = {}

    st.markdown(f"#### Pipeline: *{primary_key}*")
    card_cols = st.columns(len(METRIC_KEYS))
    for i, mk in enumerate(METRIC_KEYS):
        val = primary_metrics.get(mk, 0)
        with card_cols[i]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{METRIC_LABELS.get(mk, mk)}</div>
                    <div class="metric-value">{val:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Plotly grouped bar chart (if multiple configs)
    if len(configs) > 1:
        st.markdown("#### Ablation Comparison")
        fig = go.Figure()

        for cfg in configs:
            cfg_data = ablation_data[cfg]
            if isinstance(cfg_data, list):
                vals = {}
                for mk in METRIC_KEYS:
                    v = [float(q.get(mk, 0)) for q in cfg_data if mk in q]
                    vals[mk] = sum(v) / len(v) if v else 0.0
            else:
                vals = {mk: cfg_data.get(mk, 0) for mk in METRIC_KEYS}

            fig.add_trace(
                go.Bar(
                    name=cfg,
                    x=[METRIC_LABELS.get(mk, mk) for mk in METRIC_KEYS],
                    y=[vals[mk] for mk in METRIC_KEYS],
                    text=[f"{vals[mk]:.2f}" for mk in METRIC_KEYS],
                    textposition="auto",
                )
            )

        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            height=420,
            margin=dict(t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics table
    st.markdown("#### Detailed Metrics")
    table_rows = []
    for cfg in configs:
        cfg_data = ablation_data[cfg]
        if isinstance(cfg_data, list):
            row = {"Configuration": cfg}
            for mk in METRIC_KEYS:
                v = [float(q.get(mk, 0)) for q in cfg_data if mk in q]
                row[METRIC_LABELS.get(mk, mk)] = (
                    round(sum(v) / len(v), 4) if v else 0.0
                )
            table_rows.append(row)
        elif isinstance(cfg_data, dict):
            row = {"Configuration": cfg}
            for mk in METRIC_KEYS:
                row[METRIC_LABELS.get(mk, mk)] = round(
                    float(cfg_data.get(mk, 0)), 4
                )
            table_rows.append(row)

    if table_rows:
        st.dataframe(
            pd.DataFrame(table_rows), use_container_width=True, hide_index=True
        )


# ---------------------------------------------------------------------------
# PAGE 3: About
# ---------------------------------------------------------------------------
def page_about():
    st.markdown(
        """
    <div class="main-header">
        <h1>ArXiv Research Assistant</h1>
        <p>An agentic RAG system for querying 3,000+ ML/AI research papers using hybrid retrieval, corrective grading, and web-augmented generation.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # How It Works - 2x2 grid
    st.markdown("### How It Works")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        <div class="feature-card">
            <h4>Hybrid Retrieval</h4>
            <p>Combines dense vector search (Qdrant) with sparse BM25 retrieval, merged via Reciprocal Rank Fusion for higher recall than either alone.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div class="feature-card">
            <h4>Corrective RAG (CRAG)</h4>
            <p>Scores retrieved context confidence. Routes to web search fallback when local knowledge is insufficient, reducing hallucination.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            """
        <div class="feature-card">
            <h4>Cross-Encoder Reranking</h4>
            <p>Re-scores top candidates with a cross-encoder model for precision before passing context to the LLM.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
        <div class="feature-card">
            <h4>LangGraph Agent Loop</h4>
            <p>Stateful multi-step agent with memory, tool use, and hallucination checking built on LangGraph.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Tech Stack
    st.markdown("### Tech Stack")
    tech_data = {
        "Component": [
            "Vector Database",
            "Embeddings",
            "LLM",
            "Reranker",
            "Agent Framework",
            "Web Search",
            "Evaluation",
            "Backend",
            "Frontend",
        ],
        "Technology": [
            "Qdrant Cloud",
            "all-MiniLM-L6-v2",
            "Llama 3.3 70B via Groq",
            "ms-marco-MiniLM cross-encoder",
            "LangGraph",
            "Tavily API",
            "RAGAS",
            "FastAPI",
            "Streamlit",
        ],
    }
    st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

    # Dataset
    st.markdown("### Dataset")
    st.markdown(
        "Indexed over 3,000 ArXiv papers from **cs.AI**, **cs.CL**, and **cs.LG** "
        "categories published between 2020-2024. Papers are chunked at the abstract "
        "level with full metadata preserved."
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "Chat":
    page_chat()
elif page == "Evaluation Dashboard":
    page_evaluation()
elif page == "About":
    page_about()

# ---------------------------------------------------------------------------
# Footer (all pages)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<p class="footer-text">Built with LangGraph &middot; Qdrant &middot; Groq &middot; RAGAS</p>',
    unsafe_allow_html=True,
)
