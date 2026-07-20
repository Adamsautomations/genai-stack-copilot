"""Evaluation harness.

Scores the RAG pipeline on three axes with an LLM judge, and — critically —
scores refusal behaviour as a first-class outcome rather than an error.

    python -m src.evals.run_evals              # RAG only
    python -m src.evals.run_evals --compare    # RAG vs CAG on the same set

Groundedness is judged strictly against the passages the pipeline actually
retrieved, not against the judge's own knowledge. A judge allowed to use its
own knowledge will happily confirm a hallucination that happens to be true.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from src.config import Settings
from src.evals.questions import QUESTIONS, EvalQuestion
from src.graph.build import build_graph, initial_state
from src.llm import THINK_DYNAMIC, Usage, generate_json

RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "evals"

JUDGE_SYSTEM = """You grade a documentation assistant's answer.

You are given the question, the passages the assistant retrieved, and its \
answer.

Judge ONLY against the passages. Do not use your own knowledge of the \
libraries — if a claim is true in reality but absent from the passages, it is \
still ungrounded, because the assistant had no basis for it.

groundedness (1-5): 5 = every claim traceable to the passages. 1 = mostly \
invented.
relevance (1-5): 5 = directly answers what was asked. 1 = ignores the question.

List any claim that the passages do not support."""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "groundedness": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "relevance": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "note": {"type": "string"},
    },
    "required": ["groundedness", "relevance", "unsupported_claims", "note"],
    "propertyOrdering": ["groundedness", "relevance", "unsupported_claims", "note"],
}


def judge(settings: Settings, question: str, passages_text: str, answer: str) -> dict:
    return generate_json(
        api_key=settings.google_api_key,
        system=JUDGE_SYSTEM,
        prompt=(
            f"Question:\n{question}\n\n"
            f"Retrieved passages:\n{passages_text or '(none)'}\n\n"
            f"Assistant answer:\n{answer}"
        ),
        schema=JUDGE_SCHEMA,
        # The judge is the one classifier that keeps its reasoning — it has to
        # check each claim against the passages, which is exactly the work
        # thinking does well. A cheap judge produces cheap scores.
        thinking_budget=THINK_DYNAMIC,
        max_tokens=4096,
        label="judge",
        fallback={
            "groundedness": 0,
            "relevance": 0,
            "unsupported_claims": [],
            "note": "judge failed to return valid JSON",
        },
    )


def run_one(settings: Settings, graph, item: EvalQuestion) -> dict[str, Any]:
    started = time.perf_counter()
    final = graph.invoke(initial_state(item.question))
    elapsed = time.perf_counter() - started

    usage: Usage = final["usage"]
    passages = final.get("passages", [])
    refused = final.get("refused", False)
    answer = final.get("answer", "")

    record: dict[str, Any] = {
        "question": item.question,
        "category": item.category,
        "refused": refused,
        "refusal_reason": final.get("refusal_reason", ""),
        "answer": answer,
        "citations": len(final.get("citations", [])),
        "queries_tried": final.get("tried_queries", []),
        "latency_s": round(elapsed, 3),
        "cost_usd": round(usage.cost_usd, 6),
        "model_calls": usage.calls,
    }

    if item.category == "answerable":
        # A refusal on an answerable question is a miss; there is nothing to
        # judge for groundedness, so record it as a miss and move on.
        record["outcome"] = "miss" if refused else "answered"
        if not refused:
            passages_text = "\n\n---\n\n".join(
                p.cite(i) for i, p in enumerate(passages, start=1)
            )
            verdict = judge(settings, item.question, passages_text, answer)
            record["groundedness"] = verdict["groundedness"]
            record["relevance"] = verdict["relevance"]
            record["unsupported_claims"] = verdict["unsupported_claims"]
            record["judge_note"] = verdict["note"]
    else:
        # For both refusal categories, the correct behaviour is to refuse.
        record["outcome"] = "correct_refusal" if refused else "false_answer"

    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in records if r["category"] == "answerable"]
    answered = [r for r in answerable if r["outcome"] == "answered"]
    refusal_cases = [r for r in records if r["category"] != "answerable"]

    grounded = [r["groundedness"] for r in answered if r.get("groundedness")]
    relevant = [r["relevance"] for r in answered if r.get("relevance")]

    return {
        "totals": {
            "questions": len(records),
            "answerable": len(answerable),
            "refusal_cases": len(refusal_cases),
        },
        "answer_rate_on_answerable": (
            round(len(answered) / len(answerable), 3) if answerable else None
        ),
        "correct_refusal_rate": (
            round(
                sum(1 for r in refusal_cases if r["outcome"] == "correct_refusal")
                / len(refusal_cases),
                3,
            )
            if refusal_cases
            else None
        ),
        "false_answer_count": sum(
            1 for r in refusal_cases if r["outcome"] == "false_answer"
        ),
        "groundedness_mean": round(statistics.mean(grounded), 2) if grounded else None,
        "relevance_mean": round(statistics.mean(relevant), 2) if relevant else None,
        "answers_with_unsupported_claims": sum(
            1 for r in answered if r.get("unsupported_claims")
        ),
        "cost_usd_total": round(sum(r["cost_usd"] for r in records), 4),
        "latency_s_median": round(
            statistics.median([r["latency_s"] for r in records]), 2
        ),
    }


def main() -> int:
    settings = Settings.load()
    graph = build_graph(settings)

    print(f"Running {len(QUESTIONS)} questions through the RAG pipeline …\n")
    records: list[dict[str, Any]] = []
    for i, item in enumerate(QUESTIONS, start=1):
        record = run_one(settings, graph, item)
        records.append(record)
        mark = {
            "answered": "ok ",
            "miss": "MISS",
            "correct_refusal": "ok ",
            "false_answer": "FALSE",
        }.get(record["outcome"], "?")
        extra = ""
        if record.get("groundedness"):
            extra = f"  g={record['groundedness']} r={record['relevance']}"
        print(
            f"  [{i:>2}/{len(QUESTIONS)}] {mark:<5} {item.category:<13}"
            f" {record['latency_s']:>5.1f}s{extra}  {item.question[:52]}"
        )

    summary = summarize(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "rag_results.json"
    out.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )

    print("\n--- summary ---")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
