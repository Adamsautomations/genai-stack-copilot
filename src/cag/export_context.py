"""Export the CAG context to a standalone file.

`build_context` normally packs its prefix from `data/chunks.jsonl` — a 120 MB
file that has no business in a container image. This writes just the packed
prefix (a few hundred KB) so the deployed app can serve the CAG path without
shipping the whole corpus.

It also freezes the prefix. Rebuilding it at container start from a file whose
iteration order could drift would silently change the bytes and destroy prefix
caching; a committed artifact cannot drift.

    python -m src.cag.export_context [--context-tokens 150000]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.cag.cag import CONTEXT_EXPORT_PATH, DEFAULT_CONTEXT_TOKENS, build_context
from src.config import Settings


def main() -> int:
    tokens = DEFAULT_CONTEXT_TOKENS
    if "--context-tokens" in sys.argv:
        tokens = int(sys.argv[sys.argv.index("--context-tokens") + 1])

    settings = Settings.load()
    print(f"Packing CAG context (budget {tokens:,} tokens) …")
    context = build_context(settings, max_tokens=tokens, allow_export=False)

    CONTEXT_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTEXT_EXPORT_PATH.write_text(
        json.dumps(
            {
                "text": context.text,
                "chunk_count": context.chunk_count,
                "token_count": context.token_count,
                "sources": context.sources,
            }
        ),
        encoding="utf-8",
    )
    size_kb = CONTEXT_EXPORT_PATH.stat().st_size / 1024
    print(
        f"  {context.chunk_count} chunks, {context.token_count:,} tokens, "
        f"sources={context.sources}"
    )
    print(f"Wrote {CONTEXT_EXPORT_PATH} ({size_kb:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
