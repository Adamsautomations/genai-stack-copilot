"""RAG vs CAG on the same questions.

Runs an identical question set through both paths and reports latency, cost,
token mix, cache behaviour, and judged groundedness side by side.

Both are judged by the same judge against the same standard. The judge sees
RAG's retrieved passages, and for CAG it sees the answer alone — CAG has no
retrieval step to check against, which is itself part of the comparison: with
RAG you can audit *why* an answer was given, with CAG you largely cannot.

    python -m src.evals.compare [--questions N] [--context-tokens N]
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from src.cag.cag import answer_cag, build_context
from src.config import Settings
from src.evals.questions import by_category
from src.evals.run_evals import judge
from src.graph.build import build_graph, initial_state
from src.llm import Usage

RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "evals"


def _arg(flag: str, default: int) -> int:
    if flag in sys.argv:
        return int(sys.argv[sys.argv.index(flag) + 1])
    return default


def run_rag(settings: Settings, graph, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    final = graph.invoke(initial_state(question))
    elapsed = time.perf_counter() - started
    usage: Usage = final["usage"]
    passages = final.get("passages", [])
    passages_text = "\n\n---\n\n".join(
        p.cite(i) for i, p in enumerate(passages, start=1)
    )
    return {
        "mode": "rag",
        "answer": final.get("answer", ""),
        "refused": final.get("refused", False),
        "latency_s": round(elapsed, 2),
        "cost_usd": round(usage.cost_usd, 6),
        "prompt_tokens": usage.prompt_tokens,
        "thought_tokens": usage.thought_tokens,
        "cached_tokens": usage.cached_tokens,
        "calls": usage.calls,
        "_passages_text": passages_text,
    }


def run_cag(settings: Settings, question: str, context) -> dict[str, Any]:
    result = answer_cag(settings, question, context)
    usage = result["usage"]
    return {
        "mode": "cag",
        "answer": result["answer"],
        "refused": False,
        "latency_s": result["latency_s"],
        "cost_usd": usage["cost_usd"],
        "prompt_tokens": usage["prompt_tokens"],
        "thought_tokens": usage["thought_tokens"],
        "cached_tokens": usage["cached_tokens"],
        "calls": usage["calls"],
        "cache_hit": result["cache"]["hit"],
        "cache_hit_fraction": result["cache"]["hit_fraction"],
        "_passages_text": "",
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {
        "n": len(rows),
        "latency_s_median": round(statistics.median(r["latency_s"] for r in rows), 2),
        "cost_usd_mean": round(statistics.mean(r["cost_usd"] for r in rows), 6),
        "cost_usd_total": round(sum(r["cost_usd"] for r in rows), 5),
        "prompt_tokens_mean": round(statistics.mean(r["prompt_tokens"] for r in rows)),
        "calls_mean": round(statistics.mean(r["calls"] for r in rows), 1),
    }
    grounded = [r["groundedness"] for r in rows if r.get("groundedness")]
    if grounded:
        out["groundedness_mean"] = round(statistics.mean(grounded), 2)
    if "cache_hit" in rows[0]:
        hits = sum(1 for r in rows if r["cache_hit"])
        out["cache_hit_rate"] = round(hits / len(rows), 2)
        out["cache_hit_fraction_mean"] = round(
            statistics.mean(r["cache_hit_fraction"] for r in rows), 3
        )
    return out


def main() -> int:
    n = _arg("--questions", 6)
    context_tokens = _arg("--context-tokens", 150_000)

    settings = Settings.load()
    questions = [q.question for q in by_category("answerable")[:n]]

    print(f"Building CAG context (budget {context_tokens:,} tokens) …")
    context = build_context(settings, max_tokens=context_tokens)
    print(
        f"  {context.chunk_count} chunks, {context.token_count:,} tokens, "
        f"sources={context.sources}\n"
    )

    graph = build_graph(settings)
    rag_rows: list[dict[str, Any]] = []
    cag_rows: list[dict[str, Any]] = []

    for i, question in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {question[:60]}")

        rag = run_rag(settings, graph, question)
        if not rag["refused"]:
            v = judge(settings, question, rag["_passages_text"], rag["answer"])
            rag["groundedness"] = v["groundedness"]
            rag["relevance"] = v["relevance"]
        rag_rows.append(rag)
        print(
            f"    RAG  {rag['latency_s']:>5.1f}s  ${rag['cost_usd']:.5f}  "
            f"prompt={rag['prompt_tokens']:>6}  calls={rag['calls']}"
            f"  g={rag.get('groundedness','-')}"
        )

        cag = run_cag(settings, question, context)
        v = judge(settings, question, "", cag["answer"])
        cag["groundedness"] = v["groundedness"]
        cag["relevance"] = v["relevance"]
        cag_rows.append(cag)
        print(
            f"    CAG  {cag['latency_s']:>5.1f}s  ${cag['cost_usd']:.5f}  "
            f"prompt={cag['prompt_tokens']:>6}  cached={cag['cached_tokens']:>6}"
            f"  g={cag.get('groundedness','-')}"
        )

    for rows in (rag_rows, cag_rows):
        for r in rows:
            r.pop("_passages_text", None)

    summary = {
        "context_tokens": context.token_count,
        "context_chunks": context.chunk_count,
        "rag": aggregate(rag_rows),
        "cag": aggregate(cag_rows),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "rag_vs_cag.json"
    out.write_text(
        json.dumps({"summary": summary, "rag": rag_rows, "cag": cag_rows}, indent=2),
        encoding="utf-8",
    )

    print("\n--- summary ---")
    print(json.dumps(summary, indent=2))
    if summary["rag"].get("cost_usd_mean") and summary["cag"].get("cost_usd_mean"):
        ratio = summary["cag"]["cost_usd_mean"] / summary["rag"]["cost_usd_mean"]
        print(f"\n  CAG costs {ratio:.1f}x RAG per query on this corpus slice.")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
