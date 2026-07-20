"""Hybrid retrieval against Azure AI Search.

A query carries both `search_text` (BM25 over title/heading/content) and a
vector query (HNSW/cosine over the embedding). Azure fuses the two result
sets server-side with Reciprocal Rank Fusion — so a question that is
lexically obvious ("what is `ef_construction`") and one that is purely
semantic ("how do I stop the agent halfway") both land.

Semantic (L2) reranking is attempted only when enabled, and a failure
downgrades to plain hybrid instead of erroring: the reranker is not
available on every service tier, and a portfolio demo that dies because of
a tier limit is worse than one that quietly returns slightly coarser
ranking. The response records which path actually ran.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from src.config import Settings, TOP_K
from src.retrieval.schema import SEMANTIC_CONFIG

log = logging.getLogger(__name__)

SELECT_FIELDS = ["id", "source", "title", "url", "heading", "content"]


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk, as handed to the model."""

    id: str
    source: str
    title: str
    url: str
    heading: str
    content: str
    score: float

    def cite(self, n: int) -> str:
        """Render for the prompt with a stable citation marker."""
        where = f"{self.title} › {self.heading}" if self.heading else self.title
        return f"[{n}] {where} ({self.source})\n{self.content}"


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding

    from src.ingest.build_chunks import EMBED_MODEL

    return TextEmbedding(model_name=EMBED_MODEL)


def embed_query(text: str) -> list[float]:
    """Embed a single query. The model is loaded once and cached."""
    return next(iter(_embedder().embed([text]))).tolist()


@lru_cache(maxsize=1)
def _client(endpoint: str, index: str, key: str) -> SearchClient:
    return SearchClient(
        endpoint=endpoint, index_name=index, credential=AzureKeyCredential(key)
    )


@dataclass
class RetrievalResult:
    passages: list[Passage]
    used_semantic: bool
    """False when the semantic reranker was unavailable and we fell back."""


def retrieve(
    query: str,
    settings: Settings,
    *,
    top_k: int = TOP_K,
    source: str | None = None,
    use_semantic: bool = True,
) -> RetrievalResult:
    client = _client(settings.search_endpoint, settings.search_index, settings.search_key)
    vector_query = VectorizedQuery(
        vector=embed_query(query),
        k_nearest_neighbors=top_k * 2,  # over-fetch; RRF narrows it back down
        fields="vector",
    )

    base = {
        "search_text": query,
        "vector_queries": [vector_query],
        "select": SELECT_FIELDS,
        "top": top_k,
    }
    if source:
        base["filter"] = f"source eq '{source}'"

    used_semantic = False
    results = None

    if use_semantic:
        try:
            results = list(
                client.search(
                    **base,
                    query_type="semantic",
                    semantic_configuration_name=SEMANTIC_CONFIG,
                )
            )
            used_semantic = True
        except HttpResponseError as exc:
            log.warning(
                "semantic rerank unavailable (%s); falling back to hybrid RRF",
                exc.reason or exc.status_code,
            )

    if results is None:
        results = list(client.search(**base))

    passages = [
        Passage(
            id=r["id"],
            source=r.get("source", ""),
            title=r.get("title", ""),
            url=r.get("url", ""),
            heading=r.get("heading", ""),
            content=r.get("content", ""),
            # @search.reranker_score only exists on the semantic path.
            score=float(r.get("@search.reranker_score") or r.get("@search.score") or 0.0),
        )
        for r in results
    ]
    return RetrievalResult(passages=passages, used_semantic=used_semantic)
