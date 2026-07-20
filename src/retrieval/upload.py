"""Create the Azure AI Search index and upload the embedded chunks.

    python -m src.retrieval.upload            # create if absent, then upload
    python -m src.retrieval.upload --recreate # drop and rebuild first

Two things here are not boilerplate.

**Storage is measured, not assumed.** The free tier caps at 50 MB and there is
no reliable way to predict indexed size from raw chunk size — vectors are
stored binary, text gets an inverted index on top. So the uploader polls
`get_index_statistics()` as it goes and stops before the ceiling, reporting
actual bytes/document. Discovering the quota by hitting it mid-run leaves a
half-populated index and no idea how full it is.

**Sources are interleaved.** If the upload does stop early, a prefix of an
interleaved stream still covers every source. Uploading one source then the
next would mean truncation silently drops an entire corpus.
"""

from __future__ import annotations

import json
import sys
from itertools import zip_longest
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

from src.config import Settings
from src.retrieval.schema import build_index

CHUNKS_PATH = Path(__file__).resolve().parents[2] / "data" / "chunks.jsonl"

# Azure caps a batch at 1000 docs / 16 MB. 384-dim vectors make these
# documents heavy, so stay well under both.
BATCH_SIZE = 200

# Free tier is 50 MB. Stop here so the index stays writable for later updates.
STORAGE_BUDGET_MB = 44.0


def ensure_index(settings: Settings, *, recreate: bool = False) -> None:
    client = SearchIndexClient(
        endpoint=settings.search_endpoint,
        credential=AzureKeyCredential(settings.search_key),
    )
    if recreate:
        try:
            client.delete_index(settings.search_index)
            print(f"  dropped existing index '{settings.search_index}'")
        except ResourceNotFoundError:
            pass
    try:
        client.get_index(settings.search_index)
        print(f"  index '{settings.search_index}' already exists")
        return
    except ResourceNotFoundError:
        pass
    client.create_index(build_index(settings.search_index))
    print(f"  created index '{settings.search_index}'")


def load_interleaved() -> list[dict]:
    """Load chunks, round-robin across sources.

    Any prefix of the result is source-balanced, so an upload that stops at
    the storage ceiling still represents the whole corpus.
    """
    if not CHUNKS_PATH.exists():
        raise SystemExit(
            f"{CHUNKS_PATH} not found — run `python -m src.ingest.build_chunks` first."
        )

    by_source: dict[str, list[dict]] = {}
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            by_source.setdefault(rec["source"], []).append(rec)

    print(f"  loaded: {', '.join(f'{k}={len(v)}' for k, v in sorted(by_source.items()))}")

    interleaved: list[dict] = []
    for group in zip_longest(*(by_source[k] for k in sorted(by_source))):
        interleaved.extend(rec for rec in group if rec is not None)
    return interleaved


def storage_mb(settings: Settings) -> tuple[float, int]:
    """Index size and document count.

    Read by attribute, not `.get()`. `GetIndexStatisticsResult` *has* a `.get`
    method but it does not read model fields — it silently returns the default,
    so `stats.get("storage_size", 0)` yields 0 forever with no error. That
    turned this budget check into a no-op on the first run.

    Note the numbers lag: this endpoint reported 0 documents while the index
    was already serving all 10,250. Treat it as advisory, and confirm real
    counts with a `search_text="*"` + `include_total_count=True` query.
    """
    client = SearchIndexClient(
        endpoint=settings.search_endpoint,
        credential=AzureKeyCredential(settings.search_key),
    )
    stats = client.get_index_statistics(settings.search_index)
    return (stats.storage_size or 0) / 1024 / 1024, stats.document_count or 0


def upload(settings: Settings, docs: list[dict]) -> dict:
    client = SearchClient(
        endpoint=settings.search_endpoint,
        index_name=settings.search_index,
        credential=AzureKeyCredential(settings.search_key),
    )

    uploaded = failed = 0
    stopped_early = False

    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start : start + BATCH_SIZE]
        try:
            results = client.upload_documents(documents=batch)
        except HttpResponseError as exc:
            message = (exc.message or "").lower()
            if "quota" in message or "storage" in message:
                print(f"\n  ! storage quota reached at {uploaded} docs")
                stopped_early = True
                break
            raise SystemExit(f"upload rejected: {exc.message}") from exc

        ok = sum(1 for r in results if r.succeeded)
        uploaded += ok
        failed += len(results) - ok

        # Azure updates statistics asynchronously, so this lags slightly —
        # which is why the budget leaves headroom rather than aiming at 50.
        if (start // BATCH_SIZE) % 5 == 4:
            mb, count = storage_mb(settings)
            print(f"  {uploaded:>6}/{len(docs)} uploaded | index {mb:5.1f} MB / {count} docs")
            if mb >= STORAGE_BUDGET_MB:
                print(f"\n  reached storage budget ({STORAGE_BUDGET_MB} MB); stopping")
                stopped_early = True
                break

    mb, count = storage_mb(settings)
    return {
        "uploaded": uploaded,
        "failed": failed,
        "stopped_early": stopped_early,
        "index_mb": round(mb, 2),
        "index_docs": count,
        "bytes_per_doc": round(mb * 1024 * 1024 / count) if count else 0,
    }


def main() -> int:
    settings = Settings.load_search_only()
    recreate = "--recreate" in sys.argv

    print(f"Index: {settings.search_index} @ {settings.search_endpoint}")
    ensure_index(settings, recreate=recreate)

    docs = load_interleaved()
    print(f"\nUploading up to {len(docs)} chunks (budget {STORAGE_BUDGET_MB} MB) …")
    stats = upload(settings, docs)

    print("\n--- result ---")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    if stats["stopped_early"]:
        pct = 100 * stats["index_docs"] / len(docs)
        print(f"\n  Indexed {pct:.0f}% of the corpus — free tier storage bound.")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
