"""Graph state and the JSON schemas the structured nodes are held to."""

from __future__ import annotations

from typing import Any, TypedDict

from src.llm import Usage
from src.retrieval.search import Passage


class State(TypedDict, total=False):
    """State threaded through the graph.

    `query` is separate from `question` on purpose: the rewrite loop changes
    the search query without ever mutating what the user actually asked, so
    the synthesis step always answers the original question.
    """

    question: str
    source_filter: str | None

    # routing
    route: str  # "retrieve" | "out_of_scope"

    # retrieval
    query: str
    attempts: int
    tried_queries: list[str]
    passages: list[Passage]
    used_semantic: bool

    # grading
    verdict: str  # "answerable" | "insufficient"
    gap: str  # what the grader found missing, fed to the rewriter

    # output
    answer: str
    citations: list[dict[str, Any]]
    refused: bool
    refusal_reason: str

    # accounting
    usage: Usage
    trace: list[str]


# Gemini's responseSchema is an OpenAPI subset, not full JSON Schema:
# `additionalProperties` is unsupported, and `propertyOrdering` is honoured
# to make generation order deterministic.

ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["retrieve", "out_of_scope"],
            "description": (
                "'retrieve' if the question is about LangChain, LangGraph, "
                "LangSmith, or LlamaIndex. 'out_of_scope' for anything else."
            ),
        },
        "reason": {"type": "string", "description": "One short sentence."},
    },
    "required": ["route", "reason"],
    "propertyOrdering": ["route", "reason"],
}


GRADE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["answerable", "insufficient"],
            "description": (
                "'answerable' only if the passages contain enough to answer "
                "the question accurately. Otherwise 'insufficient'."
            ),
        },
        "gap": {
            "type": "string",
            "description": (
                "If insufficient, what specific information is missing. "
                "Empty string when answerable."
            ),
        },
    },
    "required": ["verdict", "gap"],
    "propertyOrdering": ["verdict", "gap"],
}


REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "A different search query targeting the missing information. "
                "Use vocabulary the documentation itself would use."
            ),
        }
    },
    "required": ["query"],
    "propertyOrdering": ["query"],
}
