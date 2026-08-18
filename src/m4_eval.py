from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _safe_score(value) -> float:
    """Normalize missing, NaN and infinite metric values to zero."""
    try:
        score = float(value)
        return score if math.isfinite(score) else 0.0
    except (TypeError, ValueError):
        return 0.0


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    # Implementation outline: RAGAS evaluation
    # 1. Wrap trong try/except — RAGAS cần OPENAI_API_KEY và Python 3.11+.
    # try:
    #     from ragas import evaluate
    #     from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    #     from datasets import Dataset
    #
    #     dataset = Dataset.from_dict({
    #         "question": questions, "answer": answers,
    #         "contexts": contexts, "ground_truth": ground_truths,
    #     })
    #     result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
    #                                         context_precision, context_recall])
    #     df = result.to_pandas()
    #     per_question = [EvalResult(question=row["question"], answer=row["answer"],
    #         contexts=row["contexts"], ground_truth=row["ground_truth"],
    #         faithfulness=float(row.get("faithfulness", 0.0)),
    #         answer_relevancy=float(row.get("answer_relevancy", 0.0)),
    #         context_precision=float(row.get("context_precision", 0.0)),
    #         context_recall=float(row.get("context_recall", 0.0)))
    #         for _, row in df.iterrows()]
    #     return {"faithfulness": ..., "answer_relevancy": ...,
    #             "context_precision": ..., "context_recall": ..., "per_question": [...]}
    # except Exception as e:
    #     print(f"  ⚠️  RAGAS evaluation failed: {e}")
    #     return zeros
    empty = {**{name: 0.0 for name in METRIC_NAMES}, "per_question": []}
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts and ground_truths must have equal lengths")
    if not questions:
        return empty

    from config import OPENAI_API_KEY
    if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "sk-...":
        print("  WARNING: RAGAS skipped because OPENAI_API_KEY is not configured")
        return empty
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (answer_relevancy, context_precision,
                                   context_recall, faithfulness)
        from ragas.run_config import RunConfig

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            run_config=RunConfig(
                timeout=int(os.getenv("RAGAS_TIMEOUT", "60")),
                max_retries=int(os.getenv("RAGAS_MAX_RETRIES", "2")),
                max_wait=10,
                max_workers=int(os.getenv("RAGAS_MAX_WORKERS", "4")),
            ),
        )
        frame = result.to_pandas()
        per_question = []
        for _, row in frame.iterrows():
            values = {name: _safe_score(row.get(name, 0.0)) for name in METRIC_NAMES}
            per_question.append(EvalResult(
                question=str(row["question"]), answer=str(row["answer"]),
                contexts=list(row["contexts"]), ground_truth=str(row["ground_truth"]), **values,
            ))

        aggregate = {
            name: (sum(getattr(item, name) for item in per_question) / len(per_question)
                   if per_question else 0.0)
            for name in METRIC_NAMES
        }
        return {**aggregate, "per_question": per_question}
    except Exception as exc:
        print(f"  WARNING: RAGAS evaluation failed: {exc}")
        return empty


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # Implementation outline: Diagnostic Tree failure analysis
    # 1. diagnostic_tree = {
    #        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
    #        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    #        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    #        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    #    }
    # 2. For each EvalResult: compute avg of 4 metrics, find worst_metric
    # 3. Sort by avg ascending → take bottom_n
    # 4. Return [{"question": ..., "worst_metric": ..., "score": ...,
    #             "diagnosis": ..., "suggested_fix": ...}]
    if bottom_n <= 0 or not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "Câu trả lời chứa thông tin không được context hỗ trợ.",
            "Siết system prompt, yêu cầu trích dẫn context và giảm temperature.",
        ),
        "answer_relevancy": (
            "Câu trả lời chưa tập trung vào đúng ý của câu hỏi.",
            "Cải thiện prompt trả lời và chuẩn hóa/rewrite câu hỏi trước retrieval.",
        ),
        "context_precision": (
            "Retrieval trả về nhiều chunk không liên quan.",
            "Điều chỉnh RRF/top_k, thêm reranking hoặc metadata/version filter.",
        ),
        "context_recall": (
            "Context còn thiếu thông tin cần thiết để trả lời đầy đủ.",
            "Cải thiện chunking, tăng retrieval top_k hoặc bổ sung query expansion/BM25.",
        ),
    }

    analyzed = []
    for item in eval_results:
        scores = {name: _safe_score(getattr(item, name, 0.0)) for name in METRIC_NAMES}
        worst_metric = min(METRIC_NAMES, key=lambda name: scores[name])
        average = sum(scores.values()) / len(scores)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": item.question,
            "expected": item.ground_truth,
            "got": item.answer,
            "contexts": item.contexts,
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "average_score": average,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "error_tree": (
                f"Output sai → {worst_metric} thấp ({scores[worst_metric]:.3f}) "
                f"→ {diagnosis}"
            ),
        })
    return sorted(analyzed, key=lambda item: item["average_score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {name: _safe_score(results.get(name, 0.0)) for name in METRIC_NAMES},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
