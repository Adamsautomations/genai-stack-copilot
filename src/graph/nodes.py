"""Graph nodes.

Each node takes `(settings, state)` and returns a partial state update.
`build.py` closes over settings so the graph itself stays a pure wiring
description.

The honesty rule is enforced in three places rather than one: the synthesis
system prompt forbids uncited claims, the grader gates whether synthesis runs
at all, and the refusal path is a real terminal node instead of a fallback
string. A model that is merely *asked* to be honest will drift; a graph that
cannot reach the answer node without passing a grade will not.
"""

from __future__ import annotations

import re

from src.config import MAX_RETRIEVAL_RETRIES, Settings, TOP_K
from src.graph.state import GRADE_SCHEMA, REWRITE_SCHEMA, ROUTE_SCHEMA, State
from src.llm import THINK_DYNAMIC, THINK_OFF, generate, generate_json
from src.retrieval.search import retrieve as run_retrieval

CORPUS_DESCRIPTION = (
    "the official documentation for LangChain, LangGraph, LangSmith, and LlamaIndex"
)

ROUTE_SYSTEM = f"""You route questions for a documentation assistant.

The assistant can only answer from {CORPUS_DESCRIPTION}.

Route "retrieve" when the question is about those libraries — including their \
concepts, APIs, configuration, errors, or how to build with them.
Route "out_of_scope" when it is about anything else (other frameworks, general \
programming trivia, current events, personal questions).

When genuinely unsure, prefer "retrieve" — a failed retrieval is recoverable \
and produces an honest answer; a wrong refusal is not."""

GRADE_SYSTEM = """You grade whether retrieved documentation passages are \
sufficient to answer a question accurately.

Be strict. "Answerable" means the passages contain the actual information — \
not merely that they are on a related topic. Passages that mention the right \
feature but omit the specific detail asked for are INSUFFICIENT.

If insufficient, state precisely what is missing so a different search can \
find it."""

REWRITE_SYSTEM = """You rewrite a failed documentation search query.

The previous query did not surface the information needed. Write a different \
query that targets the gap. Use the vocabulary the documentation itself would \
use — official parameter names, class names, and concept terms — rather than \
how a user might casually phrase it.

Return only the new query."""

SYNTHESIS_SYSTEM = f"""You answer questions about {CORPUS_DESCRIPTION}, using \
only the passages provided.

Rules, in order of importance:

1. Every factual claim must come from the passages. If the passages do not \
support something, do not say it.
2. Cite with bracketed numbers matching the passage numbers, like [1] or \
[2][3]. Place the citation immediately after the claim it supports.
3. If the passages only partially cover the question, answer the part they \
cover and say plainly what they do not cover. Do not fill gaps from memory.
4. Never invent API names, parameters, defaults, or version numbers. If a \
specific value is not in the passages, say it is not documented in what you \
were given.
5. Prefer showing the documented code or configuration over describing it.

Be direct. Lead with the answer, then the supporting detail."""


def _fmt_passages(state: State) -> str:
    return "\n\n---\n\n".join(
        p.cite(i) for i, p in enumerate(state.get("passages", []), start=1)
    )


def _note(state: State, message: str) -> list[str]:
    return [*state.get("trace", []), message]


# --------------------------------------------------------------------------
# supervisor
# --------------------------------------------------------------------------


def route(settings: Settings, state: State) -> State:
    """Decide whether this question can be served from the corpus at all."""
    result = generate_json(
        api_key=settings.google_api_key,
        system=ROUTE_SYSTEM,
        prompt=state["question"],
        schema=ROUTE_SCHEMA,
        thinking_budget=THINK_OFF,
        max_tokens=512,
        label="route",
        usage=state["usage"],
        fallback={"route": "retrieve", "reason": "router failed open"},
    )
    return {
        "route": result["route"],
        "query": state["question"],
        "attempts": 0,
        "tried_queries": [],
        "trace": _note(state, f"route={result['route']} ({result.get('reason','')})"),
    }


# --------------------------------------------------------------------------
# retrieval agent
# --------------------------------------------------------------------------


def retrieve(settings: Settings, state: State) -> State:
    query = state.get("query") or state["question"]
    result = run_retrieval(
        query,
        settings,
        top_k=TOP_K,
        source=state.get("source_filter"),
    )
    return {
        "passages": result.passages,
        "used_semantic": result.used_semantic,
        "tried_queries": [*state.get("tried_queries", []), query],
        "trace": _note(
            state,
            f"retrieve({query!r}) -> {len(result.passages)} passages "
            f"[{'semantic' if result.used_semantic else 'hybrid-rrf'}]",
        ),
    }


def grade(settings: Settings, state: State) -> State:
    """Gate on whether the passages actually answer the question."""
    passages = state.get("passages", [])
    if not passages:
        return {
            "verdict": "insufficient",
            "gap": "retrieval returned nothing",
            "trace": _note(state, "grade=insufficient (empty retrieval)"),
        }

    result = generate_json(
        api_key=settings.google_api_key,
        system=GRADE_SYSTEM,
        prompt=(
            f"Question: {state['question']}\n\n"
            f"Passages:\n\n{_fmt_passages(state)}"
        ),
        schema=GRADE_SCHEMA,
        thinking_budget=THINK_OFF,
        max_tokens=1024,
        label="grade",
        usage=state["usage"],
        # Fail open: if the grader itself breaks, let synthesis try. The
        # synthesis prompt still refuses to invent, so the worst case is a
        # partial answer rather than a dropped request.
        fallback={"verdict": "answerable", "gap": ""},
    )
    return {
        "verdict": result["verdict"],
        "gap": result.get("gap", ""),
        "trace": _note(state, f"grade={result['verdict']} {result.get('gap','')}".strip()),
    }


def rewrite(settings: Settings, state: State) -> State:
    """Produce a different query aimed at the gap the grader identified."""
    tried = ", ".join(repr(q) for q in state.get("tried_queries", []))
    result = generate_json(
        api_key=settings.google_api_key,
        system=REWRITE_SYSTEM,
        prompt=(
            f"Original question: {state['question']}\n"
            f"Queries already tried: {tried}\n"
            f"What was missing: {state.get('gap', 'unknown')}"
        ),
        schema=REWRITE_SCHEMA,
        thinking_budget=THINK_OFF,
        max_tokens=512,
        label="rewrite",
        usage=state["usage"],
        fallback={"query": state["question"]},
    )
    return {
        "query": result["query"],
        "attempts": state.get("attempts", 0) + 1,
        "trace": _note(state, f"rewrite -> {result['query']!r}"),
    }


# --------------------------------------------------------------------------
# synthesis agent
# --------------------------------------------------------------------------

_CITATION = re.compile(r"\[(\d+)\]")


def synthesize(settings: Settings, state: State) -> State:
    passages = state.get("passages", [])
    answer, _ = generate(
        api_key=settings.google_api_key,
        system=SYNTHESIS_SYSTEM,
        prompt=(
            f"Question: {state['question']}\n\n"
            f"Passages:\n\n{_fmt_passages(state)}"
        ),
        # The one node where reasoning earns its cost: it has to reconcile
        # several passages and attribute each claim. Classifier nodes above
        # run with thinking off.
        thinking_budget=THINK_DYNAMIC,
        max_tokens=4096,
        label="synthesize",
        usage=state["usage"],
    )

    # Only surface citations the answer actually used, in first-use order.
    used: list[int] = []
    for match in _CITATION.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= len(passages) and n not in used:
            used.append(n)

    citations = [
        {
            "n": n,
            "title": passages[n - 1].title,
            "heading": passages[n - 1].heading,
            "url": passages[n - 1].url,
            "source": passages[n - 1].source,
        }
        for n in used
    ]
    return {
        "answer": answer,
        "citations": citations,
        "refused": False,
        "trace": _note(state, f"synthesize -> {len(citations)} citations used"),
    }


def refuse(settings: Settings, state: State) -> State:
    """Terminal honest refusal.

    Deliberately not a model call: there is nothing to generate, and asking a
    model to explain why it has no information is exactly the moment it starts
    inventing some.
    """
    if state.get("route") == "out_of_scope":
        reason = "out_of_scope"
        answer = (
            f"That question is outside what I can answer. I only have "
            f"{CORPUS_DESCRIPTION}."
        )
    else:
        reason = "not_in_corpus"
        tried = state.get("tried_queries", [])
        gap = state.get("gap", "")
        answer = (
            "I could not find this in the documentation I have indexed.\n\n"
            f"I searched {len(tried)} way(s) and the passages returned did not "
            f"contain the answer."
            + (f" Specifically missing: {gap}" if gap else "")
        )
    return {
        "answer": answer,
        "citations": [],
        "refused": True,
        "refusal_reason": reason,
        "trace": _note(state, f"refuse ({reason})"),
    }


# --------------------------------------------------------------------------
# conditional edges
# --------------------------------------------------------------------------


def after_route(state: State) -> str:
    return "retrieve" if state.get("route") == "retrieve" else "refuse"


def after_grade(state: State) -> str:
    if state.get("verdict") == "answerable":
        return "synthesize"
    if state.get("attempts", 0) < MAX_RETRIEVAL_RETRIES:
        return "rewrite"
    return "refuse"
