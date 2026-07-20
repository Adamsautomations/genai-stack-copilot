"""Configuration, read once from the environment.

Fails loudly at load time when a required value is missing rather than
surfacing a confusing auth error deep inside a request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Retrieval
TOP_K = 8
"""Chunks pulled per query before grading."""

MAX_RETRIEVAL_RETRIES = 2
"""How many times the graph may re-query after a failed relevance grade."""


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    search_endpoint: str
    search_key: str
    search_index: str
    session_cost_cap_cents: float

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            google_api_key=_require("GOOGLE_API_KEY"),
            search_endpoint=_require("AZURE_SEARCH_ENDPOINT"),
            search_key=_require("AZURE_SEARCH_KEY"),
            search_index=_optional("AZURE_SEARCH_INDEX", "genai-stack-docs"),
            session_cost_cap_cents=float(_optional("SESSION_COST_CAP_CENTS", "25")),
        )

    @classmethod
    def load_search_only(cls) -> "Settings":
        """For ingestion scripts, which never call a model."""
        return cls(
            google_api_key="",
            search_endpoint=_require("AZURE_SEARCH_ENDPOINT"),
            search_key=_require("AZURE_SEARCH_KEY"),
            search_index=_optional("AZURE_SEARCH_INDEX", "genai-stack-docs"),
            session_cost_cap_cents=0.0,
        )
