"""Chunk the normalized corpus and embed it locally.

Reads `data/corpus/*.jsonl` (produced by `fetch_corpus`), splits header-first
so a chunk rarely straddles two unrelated topics, size-caps each piece, then
embeds with a local ONNX model.

Embedding locally is deliberate: it costs nothing, needs no API key, and keeps
the repo runnable by anyone who clones it. Swapping to Azure OpenAI embeddings
means changing `embed()` and the index's vector dimension — not rewriting the
pipeline.

    python -m src.ingest.build_chunks
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import Document

from src.ingest.fetch_corpus import CORPUS_DIR

OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "chunks.jsonl"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 120


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    url: str
    heading: str
    content: str


def load_documents() -> list[Document]:
    """Read every normalized source file into LlamaIndex Documents."""
    docs: list[Document] = []
    files = sorted(CORPUS_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(
            f"No corpus files in {CORPUS_DIR} — run `python -m src.ingest.fetch_corpus`."
        )

    for path in files:
        source = path.stem
        count = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                content = (rec.get("content") or "").strip()
                if not content:
                    continue
                docs.append(
                    Document(
                        text=content,
                        metadata={
                            "source": source,
                            "title": rec.get("title", ""),
                            "url": rec.get("url", ""),
                        },
                    )
                )
                count += 1
        print(f"  {source}: {count} documents")
    return docs


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """Header-split first, then size-cap. Ids are content hashes, so a re-run
    over unchanged docs produces identical ids and the upload is idempotent.

    The size-capping pass runs on bare text via `split_text`, not through
    `get_nodes_from_documents`. The latter is metadata-aware: it reserves part
    of the chunk budget for the serialized metadata, and MarkdownNodeParser's
    accumulated header trail can exceed the budget on its own (deeply nested
    headings produced a 1830-char metadata string against an 800-token chunk).
    Metadata is only needed for citations, never for splitting, so it is
    reattached afterwards instead of competing for space.
    """
    md_parser = MarkdownNodeParser()
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks: list[Chunk] = []
    seen: set[str] = set()

    for node in md_parser.get_nodes_from_documents(docs):
        meta = node.metadata
        heading = " › ".join(
            str(v) for k, v in meta.items() if k.lower().startswith("header") and v
        )
        if len(heading) > 200:  # keep the breadcrumb readable in a citation
            heading = heading[:197] + "…"

        for piece in splitter.split_text(node.get_content()):
            content = piece.strip()
            if len(content) < MIN_CHUNK_CHARS:
                continue

            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
            if digest in seen:  # boilerplate repeated verbatim across pages
                continue
            seen.add(digest)

            chunks.append(
                Chunk(
                    id=digest,
                    source=meta.get("source", "unknown"),
                    title=meta.get("title", ""),
                    url=meta.get("url", ""),
                    heading=heading,
                    content=content,
                )
            )
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    """Embed with a local ONNX model. Downloads the model on first run."""
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=EMBED_MODEL)
    out: list[list[float]] = []
    for i, vec in enumerate(model.embed(texts, batch_size=64), start=1):
        out.append(vec.tolist())
        if i % 1000 == 0:
            print(f"    embedded {i}/{len(texts)}")
    return out


def main() -> int:
    print("Loading documents …")
    docs = load_documents()

    print(f"\nChunking {len(docs)} documents …")
    chunks = chunk_documents(docs)
    print(f"  {len(chunks)} chunks after dedupe")

    print(f"\nEmbedding with {EMBED_MODEL} ({EMBED_DIM}d) …")
    vectors = embed([c.content for c in chunks])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for chunk, vector in zip(chunks, vectors, strict=True):
            record = asdict(chunk)
            record["vector"] = vector
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nWrote {len(chunks)} chunks to {OUT_PATH} ({size_mb:.1f} MB)")

    by_source: dict[str, int] = {}
    for c in chunks:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    for source, n in sorted(by_source.items()):
        print(f"  {source}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
