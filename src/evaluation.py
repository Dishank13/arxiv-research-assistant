"""
Phase 7 — Evaluation Module using RAGAS
Runs RAGAS evaluation on the agent pipeline with ablation studies.
Falls back to manual text-overlap metrics if RAGAS is unavailable.
"""

import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "eval", "test_questions.json")
ABLATION_RESULTS_PATH = os.path.join(PROJECT_ROOT, "eval", "ablation_results.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_test_questions(path: str = TEST_QUESTIONS_PATH) -> list[dict]:
    """Load test questions from the JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _simple_overlap(prediction: str, reference: str) -> float:
    """Compute a simple token-level F1 overlap score as a fallback metric."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _manual_evaluation(results: list[dict]) -> dict:
    """
    Fallback evaluation when RAGAS is not available.
    Computes simple text-overlap-based proxies for the four RAGAS metrics.
    """
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []

    for r in results:
        answer = r.get("answer", "")
        ground_truth = r.get("ground_truth", "")
        contexts = r.get("contexts", [])
        context_text = " ".join(contexts)

        # Faithfulness proxy: overlap between answer and contexts
        faithfulness_scores.append(_simple_overlap(answer, context_text) if context_text else 0.0)

        # Answer relevancy proxy: overlap between answer and question
        relevancy_scores.append(_simple_overlap(answer, r.get("question", "")))

        # Context precision proxy: overlap between contexts and ground truth
        precision_scores.append(_simple_overlap(context_text, ground_truth) if context_text else 0.0)

        # Context recall proxy: overlap between contexts and ground truth (same as above, different name)
        recall_scores.append(_simple_overlap(context_text, ground_truth) if context_text else 0.0)

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "faithfulness": round(_mean(faithfulness_scores), 4),
        "answer_relevancy": round(_mean(relevancy_scores), 4),
        "context_precision": round(_mean(precision_scores), 4),
        "context_recall": round(_mean(recall_scores), 4),
        "_method": "manual_overlap_fallback",
    }


# ---------------------------------------------------------------------------
# Run pipeline on questions
# ---------------------------------------------------------------------------
def _run_pipeline_on_questions(
    questions: list[dict],
    retrieval_mode: str = "full",
) -> list[dict]:
    """
    Run the agent pipeline on a list of question dicts.
    Returns a list of result dicts containing question, answer, contexts, ground_truth.
    """
    from src.agent import query_agent  # noqa: E402 – imported here to avoid circular deps

    results = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['question'][:80]}...")
        try:
            response = query_agent(q["question"], retrieval_mode=retrieval_mode)
            answer = response.get("answer", "") if isinstance(response, dict) else str(response)
            # Extract contexts from documents (list of dicts with 'text' key)
            documents = response.get("documents", []) if isinstance(response, dict) else []
            contexts = []
            for doc in documents:
                if isinstance(doc, dict) and doc.get("text"):
                    contexts.append(doc["text"])
                elif isinstance(doc, str):
                    contexts.append(doc)
            # If no document texts, use the context string
            if not contexts and response.get("context"):
                contexts = [response["context"]]
        except Exception as e:
            print(f"    ⚠ Agent error: {e}")
            answer = ""
            contexts = []

        results.append({
            "question": q["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": q["ground_truth"],
        })

        # Rate limiting for Groq API
        time.sleep(2)

    return results


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------
def _ragas_evaluate(results: list[dict]) -> dict:
    """
    Run RAGAS evaluation on collected results.
    Falls back to manual evaluation if RAGAS is unavailable or errors out.
    """
    try:
        # --- Attempt RAGAS v0.2+ API ---
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from ragas import EvaluationDataset, SingleTurnSample
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0,
            )

            samples = []
            for r in results:
                samples.append(
                    SingleTurnSample(
                        user_input=r["question"],
                        response=r["answer"],
                        retrieved_contexts=r["contexts"],
                        reference=r["ground_truth"],
                    )
                )

            dataset = EvaluationDataset(samples=samples)
            eval_result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=llm,
            )

            scores = eval_result.to_pandas().mean(numeric_only=True).to_dict()
            return {
                "faithfulness": round(scores.get("faithfulness", 0), 4),
                "answer_relevancy": round(scores.get("answer_relevancy", 0), 4),
                "context_precision": round(scores.get("context_precision", 0), 4),
                "context_recall": round(scores.get("context_recall", 0), 4),
                "_method": "ragas_v0.2+",
            }

        except ImportError:
            # --- Attempt RAGAS v0.1.x API ---
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0,
            )

            eval_data = {
                "question": [r["question"] for r in results],
                "answer": [r["answer"] for r in results],
                "contexts": [r["contexts"] for r in results],
                "ground_truth": [r["ground_truth"] for r in results],
            }
            dataset = Dataset.from_dict(eval_data)

            eval_result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=llm,
            )

            return {
                "faithfulness": round(eval_result.get("faithfulness", 0), 4),
                "answer_relevancy": round(eval_result.get("answer_relevancy", 0), 4),
                "context_precision": round(eval_result.get("context_precision", 0), 4),
                "context_recall": round(eval_result.get("context_recall", 0), 4),
                "_method": "ragas_v0.1.x",
            }

    except Exception as e:
        print(f"  ⚠ RAGAS evaluation failed ({type(e).__name__}: {e})")
        print("  → Falling back to manual text-overlap evaluation.")
        return _manual_evaluation(results)


# ---------------------------------------------------------------------------
# Ablation studies
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = {
    "dense_only": "dense_only",
    "hybrid_no_rerank": "hybrid_no_rerank",
    "full_pipeline": "full",
}


def _run_ablation(questions: list[dict], n_subset: int = 10) -> dict:
    """
    Run ablation studies across 3 retrieval configurations on a subset of questions.
    """
    subset = questions[:n_subset]
    ablation_results = {}

    for name, mode in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Ablation: {name} (retrieval_mode={mode})")
        print(f"{'='*60}")
        results = _run_pipeline_on_questions(subset, retrieval_mode=mode)
        metrics = _ragas_evaluate(results)
        # Remove internal _method key from saved metrics
        method = metrics.pop("_method", "unknown")
        print(f"  → {name} metrics ({method}): {metrics}")
        ablation_results[name] = metrics

    return ablation_results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def _save_results(results: dict, path: str = ABLATION_RESULTS_PATH) -> None:
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to {path}")


# ---------------------------------------------------------------------------
# Main evaluation functions
# ---------------------------------------------------------------------------
def run_evaluation() -> dict:
    """
    Full evaluation pipeline:
    1. Load all 25 test questions
    2. Run full pipeline evaluation on all questions
    3. Run ablation studies on a 10-question subset
    4. Save and return results
    """
    print("=" * 70)
    print("  RAGAS Evaluation — Full Run")
    print("=" * 70)

    questions = load_test_questions()
    print(f"\n📋 Loaded {len(questions)} test questions.\n")

    # --- Full pipeline evaluation on ALL questions ---
    print("\n▶ Running full pipeline on all questions...")
    full_results = _run_pipeline_on_questions(questions, retrieval_mode="full")
    full_metrics = _ragas_evaluate(full_results)
    method = full_metrics.pop("_method", "unknown")
    print(f"  → Full pipeline metrics ({method}): {full_metrics}")

    # --- Ablation studies on subset ---
    print("\n▶ Running ablation studies (10-question subset)...")
    ablation = _run_ablation(questions, n_subset=10)

    # Override full_pipeline in ablation with the full-question results
    ablation["full_pipeline"] = full_metrics

    output = {
        "timestamp": datetime.now().isoformat(),
        **ablation,
    }

    _save_results(output)

    print("\n" + "=" * 70)
    print("  Phase 7 complete — ready for Phase 8")
    print("=" * 70)

    return output


def run_quick_evaluation(n_questions: int = 5) -> dict:
    """
    Quick evaluation on a small subset of questions for faster testing.
    """
    print("=" * 70)
    print(f"  RAGAS Evaluation — Quick Run ({n_questions} questions)")
    print("=" * 70)

    questions = load_test_questions()[:n_questions]
    print(f"\n📋 Using {len(questions)} test questions.\n")

    results_by_config = {}
    for name, mode in ABLATION_CONFIGS.items():
        print(f"\n▶ Running {name} (retrieval_mode={mode})...")
        results = _run_pipeline_on_questions(questions, retrieval_mode=mode)
        metrics = _ragas_evaluate(results)
        method = metrics.pop("_method", "unknown")
        print(f"  → {name} metrics ({method}): {metrics}")
        results_by_config[name] = metrics

    output = {
        "timestamp": datetime.now().isoformat(),
        **results_by_config,
    }

    _save_results(output)
    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG pipeline.")
    parser.add_argument(
        "--quick",
        type=int,
        default=0,
        metavar="N",
        help="Run a quick evaluation on N questions (default: full evaluation).",
    )
    args = parser.parse_args()

    if args.quick > 0:
        results = run_quick_evaluation(n_questions=args.quick)
    else:
        results = run_evaluation()

    print("\n📊 Final Results:")
    print(json.dumps(results, indent=2))
    print("\nPhase 7 complete — ready for Phase 8")
