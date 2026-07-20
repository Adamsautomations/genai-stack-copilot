"""The evaluation question set.

Three categories, because a retrieval system has three ways to be wrong and
only one of them is "gave a bad answer":

* `answerable`     — the corpus covers it; a refusal here is a miss.
* `out_of_scope`   — nothing to do with the corpus; an answer here is a
                     hallucination the router should have prevented.
* `not_in_corpus`  — sounds in-domain but isn't covered; an answer here is
                     the expensive failure, because it will look plausible.

Most RAG evaluations only measure the first category, which is how systems
ship that answer everything confidently.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    category: str  # answerable | out_of_scope | not_in_corpus


QUESTIONS: tuple[EvalQuestion, ...] = (
    # --- LangGraph ---------------------------------------------------------
    EvalQuestion("How do I add persistence to a LangGraph graph?", "answerable"),
    EvalQuestion("What is a checkpointer in LangGraph?", "answerable"),
    EvalQuestion("How do I stream output from a LangGraph graph?", "answerable"),
    EvalQuestion("What is the difference between a node and an edge in LangGraph?", "answerable"),
    EvalQuestion("How do subgraphs work in LangGraph?", "answerable"),
    EvalQuestion("How do I add memory to a LangGraph agent?", "answerable"),
    EvalQuestion("What are conditional edges in LangGraph?", "answerable"),
    EvalQuestion("How do I pause a LangGraph run for human input?", "answerable"),
    # --- LangChain / LangSmith --------------------------------------------
    EvalQuestion("What is LangSmith used for?", "answerable"),
    EvalQuestion("How do I add tracing to a LangChain application?", "answerable"),
    EvalQuestion("What is a tool in LangChain and how do I define one?", "answerable"),
    # --- LlamaIndex --------------------------------------------------------
    EvalQuestion("What is a VectorStoreIndex in LlamaIndex?", "answerable"),
    EvalQuestion("How does LlamaIndex split documents into nodes?", "answerable"),
    EvalQuestion("What is a query engine in LlamaIndex?", "answerable"),
    EvalQuestion("What is a node parser in LlamaIndex?", "answerable"),
    # --- clearly outside the corpus ---------------------------------------
    EvalQuestion("What is the capital of France?", "out_of_scope"),
    EvalQuestion("How do I center a div with CSS flexbox?", "out_of_scope"),
    EvalQuestion("Who won the football World Cup in 2022?", "out_of_scope"),
    EvalQuestion("What is my current account balance?", "out_of_scope"),
    # --- plausible but not covered ----------------------------------------
    EvalQuestion(
        "What is the exact per-token price of Claude Opus 4.8 in euros?",
        "not_in_corpus",
    ),
    EvalQuestion(
        "What is the default TCP keepalive timeout for LangGraph's Postgres "
        "checkpointer on FreeBSD?",
        "not_in_corpus",
    ),
    EvalQuestion(
        "Which LangGraph release removed the `graph.step_timeout` attribute?",
        "not_in_corpus",
    ),
)


def by_category(category: str) -> tuple[EvalQuestion, ...]:
    return tuple(q for q in QUESTIONS if q.category == category)
