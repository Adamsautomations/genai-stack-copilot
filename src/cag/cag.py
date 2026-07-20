"""Cache-Augmented Generation on Gemini.

The alternative to retrieving per query: put the corpus in the context *once*
and answer every question against it. No retriever, no ranking, no grader.

Two constraints decide whether that is a good idea, and both were measured on
this deployment rather than assumed:

1. **The corpus must fit the context window.** This project's full corpus does
   not — 10k+ chunks is far past what is sensible to send per call — so CAG
   here runs over a bounded slice. That is the finding, not a workaround: CAG
   does not degrade gracefully as a corpus grows, it stops being applicable.

2. **Caching is what makes it cheap, and here caching is best-effort.**
   Vertex *Express* keys cannot reach the explicit `cachedContents` API (it
   404s — Express hides the project/location path the cache API is addressed
   by). That leaves implicit caching, which is real but not guaranteed:

       call 1: prompt=27011  cached=0
       call 2: prompt=27011  cached=26586   (98.4% hit)
       call 3: prompt=27011  cached=0

   Byte-identical prefix each time. So the cost of a CAG query here is a
   distribution, not a number, and this module reports the hit *rate* across
   a run rather than a single flattering measurement.

The corpus is placed in `systemInstruction` because that is the most stable
part of the request — the question varies, the corpus must not.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import Settings
from src.llm import (
    MODEL_SYNTHESIS,
    THINK_DYNAMIC,
    Usage,
    count_tokens,
    generate,
)

CHUNKS_PATH = Path(__file__).resolve().parents[2] / "data" / "chunks.jsonl"

DEFAULT_CONTEXT_TOKENS = 150_000

CAG_SYSTEM_PREFIX = """You answer questions about the LangChain, LangGraph, \
LangSmith, and LlamaIndex documentation, using only the documentation below.

Rules:
1. Every factual claim must come from the documentation below.
2. Cite the section title you drew from, in brackets, after the claim.
3. If the documentation below does not cover the question, say so plainly. \
Do not answer from memory.
4. Never invent API names, parameters, defaults, or version numbers.

Be direct. Lead with the answer.

=== DOCUMENTATION ===

"""


@dataclass
class CagContext:
    """A frozen corpus slice. Any change to `text` breaks prefix caching."""

    text: str
    chunk_count: int
    token_count: int
    sources: dict[str, int]


def build_context(
    settings: Settings,
    *,
    max_tokens: int = DEFAULT_CONTEXT_TOKENS,
    source: str | None = None,
) -> CagContext:
    """Pack chunks into one stable block under a token budget.

    Chunks are taken in corpus order, not relevance order — there is no query
    yet, which is the entire premise of CAG. Order is deterministic so the
    prefix is byte-identical across runs and can actually cache.
    """
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"{CHUNKS_PATH} not found — build the corpus first.")

    parts: list[str] = []
    sources: dict[str, int] = {}
    running_chars = 0
    # ~3.6 chars/token for technical English. Estimating while packing avoids
    # a network round-trip per chunk; the exact count is verified once below.
    char_budget = max_tokens * 3.6

    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if source and rec["source"] != source:
                continue

            where = rec["title"]
            if rec.get("heading"):
                where = f"{where} › {rec['heading']}"
            block = f"## {where}\n{rec['content']}"

            if running_chars + len(block) > char_budget:
                break
            parts.append(block)
            running_chars += len(block)
            sources[rec["source"]] = sources.get(rec["source"], 0) + 1

    text = CAG_SYSTEM_PREFIX + "\n\n".join(parts)
    exact = count_tokens(settings.google_api_key, MODEL_SYNTHESIS, text)

    return CagContext(
        text=text, chunk_count=len(parts), token_count=exact, sources=sources
    )


def answer_cag(
    settings: Settings,
    question: str,
    context: CagContext,
) -> dict[str, Any]:
    """Answer from the in-context corpus. Reports whether the prefix cached."""
    usage = Usage()
    started = time.perf_counter()

    answer, payload = generate(
        api_key=settings.google_api_key,
        system=context.text,  # stable prefix — the whole point
        prompt=question,
        model=MODEL_SYNTHESIS,
        thinking_budget=THINK_DYNAMIC,
        max_tokens=4096,
        label="cag",
        usage=usage,
    )
    elapsed = time.perf_counter() - started

    meta = payload.get("usageMetadata") or {}
    prompt_tokens = int(meta.get("promptTokenCount") or 0)
    cached = int(meta.get("cachedContentTokenCount") or 0)

    return {
        "question": question,
        "answer": answer,
        "mode": "cag",
        "latency_s": round(elapsed, 3),
        "cache": {
            "cached_tokens": cached,
            "prompt_tokens": prompt_tokens,
            "hit": cached > 0,
            "hit_fraction": round(cached / prompt_tokens, 4) if prompt_tokens else 0.0,
            "note": "implicit caching is best-effort on this deployment",
        },
        "context": {
            "chunks": context.chunk_count,
            "tokens": context.token_count,
            "sources": context.sources,
        },
        "citations": [],  # CAG cites inline by section title, not by index
        "refused": False,
        "usage": usage.summary(),
    }
