"""Fetch the documentation corpus and normalize it.

Two source shapes, one output shape.

Upstream doc layouts are unstable — LangGraph stopped shipping markdown in
its repo and now publishes an `llms-full.txt`; LlamaIndex moved its docs to
`docs/src/content`; Microsoft removed Azure Search docs from `azure-docs`
entirely. Rather than let that churn leak into the chunker, every source is
normalized here into `data/corpus/<name>.jsonl` with a fixed record shape:

    {"title": str, "url": str, "content": str}

Adding a source means writing one fetcher, not touching anything downstream.

    python -m src.ingest.fetch_corpus [--force]
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"

# `llms-full.txt` concatenates every page as:
#     # <Title>
#     Source: <url>
#
#     <body>
_LLMS_DOC = re.compile(
    r"^#[ \t]+(?P<title>.+?)[ \t]*\nSource:[ \t]*(?P<url>\S+)[ \t]*\n(?P<body>.*?)(?=^#[ \t]+.+?\nSource:|\Z)",
    re.MULTILINE | re.DOTALL,
)

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TITLE_IN_FM = re.compile(r"^title:\s*(.+?)\s*$", re.MULTILINE)
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class LlmsFullSource:
    """A site publishing the llms-full.txt convention."""

    name: str
    url: str
    exclude_url_parts: tuple[str, ...] = ()
    """Drop documents whose Source URL contains any of these."""
    min_chars: int = 400
    """Skip stubs — API-reference entries are often just a URL and a verb."""


@dataclass(frozen=True)
class GitSource:
    """A repo whose docs are markdown on disk."""

    name: str
    repo: str
    sparse_path: str
    base_url: str
    url_trim: str = ""
    """Prefix removed from the relative path when building the citation URL."""


LLMS_SOURCES: tuple[LlmsFullSource, ...] = (
    LlmsFullSource(
        name="langchain",
        url="https://docs.langchain.com/llms-full.txt",
        # /api-reference/ entries are generated OpenAPI stubs: a URL, an HTTP
        # verb, and little else. They inflate the index without being
        # answerable, so they are dropped rather than indexed and ignored.
        exclude_url_parts=("/api-reference/",),
    ),
)

GIT_SOURCES: tuple[GitSource, ...] = (
    GitSource(
        name="llamaindex",
        repo="run-llama/llama_index",
        sparse_path="docs/src/content/docs",
        base_url="https://docs.llamaindex.ai/en/stable/",
        url_trim="docs/src/content/docs/",
    ),
)

ALL_SOURCE_NAMES = tuple(s.name for s in LLMS_SOURCES) + tuple(s.name for s in GIT_SOURCES)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{result.stderr.strip()}")


def _write_jsonl(name: str, records: list[dict]) -> Path:
    out = CORPUS_DIR / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


# --------------------------------------------------------------------------
# llms-full.txt sources
# --------------------------------------------------------------------------


def fetch_llms_full(source: LlmsFullSource, *, force: bool) -> int:
    out = CORPUS_DIR / f"{source.name}.jsonl"
    if out.exists() and not force:
        print(f"  {source.name}: already normalized, skipping (--force to refetch)")
        return sum(1 for _ in out.open(encoding="utf-8"))

    print(f"  {source.name}: downloading {source.url} …")
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        response = client.get(source.url)
        response.raise_for_status()
    text = response.text
    print(f"  {source.name}: {len(text) / 1024 / 1024:.1f} MB downloaded, parsing …")

    records: list[dict] = []
    skipped_excluded = skipped_short = 0
    for match in _LLMS_DOC.finditer(text):
        url = match.group("url").strip()
        if any(part in url for part in source.exclude_url_parts):
            skipped_excluded += 1
            continue
        body = match.group("body").strip()
        if len(body) < source.min_chars:
            skipped_short += 1
            continue
        records.append(
            {"title": match.group("title").strip(), "url": url, "content": body}
        )

    _write_jsonl(source.name, records)
    print(
        f"  {source.name}: {len(records)} documents "
        f"({skipped_excluded} excluded by URL, {skipped_short} too short)"
    )
    return len(records)


# --------------------------------------------------------------------------
# git sources
# --------------------------------------------------------------------------


def _derive_title(body: str, fm_title: str | None, path: Path) -> str:
    if fm_title:
        return fm_title
    h1 = _H1.search(body)
    if h1:
        return h1.group(1).strip()
    return path.stem.replace("-", " ").replace("_", " ").title()


def fetch_git(source: GitSource, *, force: bool) -> int:
    out = CORPUS_DIR / f"{source.name}.jsonl"
    if out.exists() and not force:
        print(f"  {source.name}: already normalized, skipping (--force to refetch)")
        return sum(1 for _ in out.open(encoding="utf-8"))

    clone_dir = CORPUS_DIR / f"_{source.name}_repo"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    print(f"  {source.name}: sparse-cloning {source.repo}/{source.sparse_path} …")
    _run(
        [
            "git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
            f"https://github.com/{source.repo}.git", str(clone_dir),
        ]
    )
    _run(["git", "sparse-checkout", "set", source.sparse_path], cwd=clone_dir)

    doc_root = clone_dir / source.sparse_path
    if not doc_root.exists():
        raise RuntimeError(f"{source.name}: {source.sparse_path} missing after clone")

    records: list[dict] = []
    for path in sorted([*doc_root.rglob("*.md"), *doc_root.rglob("*.mdx")]):
        if any(p in {"_static", "images", "assets"} for p in path.parts):
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        fm_title = None
        fm = _FRONTMATTER.match(raw)
        if fm:
            m = _TITLE_IN_FM.search(fm.group(1))
            fm_title = m.group(1).strip().strip("\"'") if m else None
            raw = raw[fm.end():]

        body = raw.strip()
        if len(body) < 400:
            continue

        rel = path.relative_to(clone_dir).as_posix()
        slug = re.sub(r"\.mdx?$", "", rel.removeprefix(source.url_trim))
        slug = re.sub(r"/index$", "/", slug)

        records.append(
            {
                "title": _derive_title(body, fm_title, path),
                "url": f"{source.base_url}{slug}",
                "content": body,
            }
        )

    _write_jsonl(source.name, records)
    shutil.rmtree(clone_dir, ignore_errors=True)  # keep only the normalized output
    print(f"  {source.name}: {len(records)} documents")
    return len(records)


def main() -> int:
    force = "--force" in sys.argv
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Normalizing corpus into {CORPUS_DIR}\n")

    total = 0
    for llms_source in LLMS_SOURCES:
        total += fetch_llms_full(llms_source, force=force)
    for git_source in GIT_SOURCES:
        total += fetch_git(git_source, force=force)

    print(f"\nTotal: {total} documents across {len(ALL_SOURCE_NAMES)} sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
