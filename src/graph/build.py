"""Graph wiring.

    route ──┬─► retrieve ─► grade ──┬─► synthesize ─► END
            │        ▲              │
            │        └── rewrite ◄──┤   (insufficient, attempts < max)
            │                       │
            └────────────────────────► refuse ─► END
                (out of scope)          (insufficient, retries exhausted)

The cycle is the point. A single-shot retrieve→generate pipeline answers
confidently from whatever it happened to pull; putting a grader between
retrieval and generation means a bad first search costs one cheap extra call
instead of producing a wrong answer.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from src.config import Settings
from src.graph import nodes
from src.graph.state import State
from src.llm import Usage


def build_graph(settings: Settings):
    """Compile the graph with `settings` bound into every node."""
    g = StateGraph(State)

    g.add_node("route", partial(nodes.route, settings))
    g.add_node("retrieve", partial(nodes.retrieve, settings))
    g.add_node("grade", partial(nodes.grade, settings))
    g.add_node("rewrite", partial(nodes.rewrite, settings))
    g.add_node("synthesize", partial(nodes.synthesize, settings))
    g.add_node("refuse", partial(nodes.refuse, settings))

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route", nodes.after_route, {"retrieve": "retrieve", "refuse": "refuse"}
    )
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        nodes.after_grade,
        {"synthesize": "synthesize", "rewrite": "rewrite", "refuse": "refuse"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("synthesize", END)
    g.add_edge("refuse", END)

    return g.compile()


def initial_state(question: str, *, source_filter: str | None = None) -> State:
    return {
        "question": question,
        "source_filter": source_filter,
        "query": question,
        "attempts": 0,
        "tried_queries": [],
        "passages": [],
        "usage": Usage(),
        "trace": [],
    }


def answer(settings: Settings, question: str, *, source_filter: str | None = None) -> dict:
    """Run one question end to end and return a serializable result."""
    graph = build_graph(settings)
    final: State = graph.invoke(initial_state(question, source_filter=source_filter))

    usage: Usage = final["usage"]
    return {
        "question": question,
        "answer": final.get("answer", ""),
        "citations": final.get("citations", []),
        "refused": final.get("refused", False),
        "refusal_reason": final.get("refusal_reason", ""),
        "retrieval": {
            "queries_tried": final.get("tried_queries", []),
            "passages_returned": len(final.get("passages", [])),
            "used_semantic_rerank": final.get("used_semantic", False),
        },
        "usage": usage.summary(),
        "trace": final.get("trace", []),
    }
