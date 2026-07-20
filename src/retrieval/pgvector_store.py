"""The same hybrid retrieval, rebuilt on Postgres + pgvector.

Azure AI Search does three things server-side that make it a one-line query:
HNSW vector search, BM25 keyword search, and Reciprocal Rank Fusion over the
two. Postgres has the first two — `pgvector` for HNSW, `tsvector`/`ts_rank` for
keyword — but nothing that fuses them. So the fusion is written out here, which
is the whole point of the exercise: it makes explicit the step Azure hides, and
that step is where hybrid retrieval actually earns its keep.

The embeddings are reused, not recomputed. `data/chunks.jsonl` already carries
384-dim `bge-small-en-v1.5` vectors, so both engines are searching *identical*
vectors and any difference in results is a property of the index and the
fusion, not of the embedding.

    python -m src.retrieval.pgvector_store --load     # create + index + load
    python -m src.retrieval.pgvector_store --query "how do I stream tokens"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "data" / "chunks.jsonl"

EMBED_DIM = 384
# RRF's damping constant. 60 is the value from the original Cormack et al.
# paper and what Azure AI Search uses, so keeping it makes the two engines
# comparable rather than merely similar.
RRF_K = 60


def wsl_host() -> str | None:
    """Resolve the WSL VM's IP.

    WSL reassigns this on restart, so it is looked up rather than stored.
    `localhost` usually works via WSL's port relay, but the relay silently
    stops forwarding often enough that having the real address as a fallback
    saves a confusing debugging session.
    """
    try:
        out = subprocess.run(
            ["wsl", "-d", "Ubuntu-26.04", "-u", "root", "--", "hostname", "-I"],
            capture_output=True, text=True, timeout=25,
        )
        return out.stdout.strip().split()[0] or None
    except Exception:
        return None


def dsn() -> str:
    """Connection string, overridable by `PGVECTOR_DSN` / `PGVECTOR_HOST`.

    The default credential is deliberately a hardcoded throwaway: it belongs to
    a benchmark database inside a local WSL VM that is not reachable from
    outside this machine and holds nothing but public documentation chunks.
    Putting it in `.env` would imply it is a secret worth protecting and would
    stop the benchmark being reproducible by cloning. Anything real goes
    through `PGVECTOR_DSN`.
    """
    if os.getenv("PGVECTOR_DSN"):
        return os.environ["PGVECTOR_DSN"]
    host = os.getenv("PGVECTOR_HOST") or wsl_host() or "localhost"
    return (
        f"host={host} port=5432 dbname=ragbench "
        "user=postgres password=ragbench connect_timeout=10"
    )


@lru_cache(maxsize=1)
def _connect():
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(dsn(), autocommit=True)
    register_vector(conn)
    return conn


DDL = f"""
CREATE TABLE IF NOT EXISTS chunks (
    id       text PRIMARY KEY,
    source   text NOT NULL,
    title    text,
    url      text,
    heading  text,
    content  text NOT NULL,
    embedding vector({EMBED_DIM}) NOT NULL,
    -- Generated rather than maintained by the loader: it cannot drift out of
    -- sync with `content`, which is the usual bug with a denormalised
    -- search column.
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')),   'A') ||
        setweight(to_tsvector('english', coalesce(heading, '')), 'B') ||
        setweight(to_tsvector('english', content),               'C')
    ) STORED
);
"""

# m/ef_construction are pgvector's defaults, stated explicitly because they are
# the recall-vs-build-time knob an interviewer will ask about. Higher m =
# denser graph = better recall, bigger index, slower build.
INDEXES = [
    "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks "
    "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
    "CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv)",
]


@dataclass(frozen=True)
class Hit:
    id: str
    source: str
    title: str
    url: str
    heading: str
    content: str
    score: float
    vector_rank: int | None
    text_rank: int | None


def load() -> int:
    """Create the schema and bulk-load the chunks that already have vectors."""
    if not CHUNKS.exists():
        raise SystemExit(f"missing {CHUNKS} — run the ingest step first")

    conn = _connect()
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(DDL)

    started = time.perf_counter()
    n = 0
    # COPY rather than INSERT: 10k rows of 384 floats is ~30MB on the wire and
    # a per-row round trip turns a 5-second load into minutes.
    with conn.cursor().copy(
        "COPY chunks (id, source, title, url, heading, content, embedding) "
        "FROM STDIN WITH (FORMAT BINARY)"
    ) as copy:
        copy.set_types(["text", "text", "text", "text", "text", "text", "vector"])
        with CHUNKS.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                copy.write_row((
                    r["id"], r.get("source", ""), r.get("title", ""),
                    r.get("url", ""), r.get("heading", ""), r["content"],
                    r["vector"],
                ))
                n += 1
    copied = time.perf_counter() - started

    print(f"  copied {n:,} rows in {copied:.1f}s")
    for stmt in INDEXES:
        t0 = time.perf_counter()
        conn.execute(stmt)
        label = "hnsw" if "hnsw" in stmt else "gin"
        print(f"  built {label} index in {time.perf_counter() - t0:.1f}s")

    conn.execute("ANALYZE chunks")
    size = conn.execute(
        "SELECT pg_size_pretty(pg_total_relation_size('chunks'))"
    ).fetchone()[0]
    print(f"  total relation size: {size}")
    return n


# Vector and keyword candidates are each ranked independently, then fused by
# summing 1/(k + rank). RRF deliberately ignores the raw scores: cosine
# distance and ts_rank are not on comparable scales, and any attempt to
# normalise them into a weighted sum needs a tuning constant per corpus.
SEARCH_SQL = """
WITH vec AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(q)s::vector) AS rank
    FROM chunks
    ORDER BY embedding <=> %(q)s::vector
    LIMIT %(cand)s
),
txt AS (
    SELECT id, ROW_NUMBER() OVER (
        ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', %(text)s)) DESC
    ) AS rank
    FROM chunks
    WHERE tsv @@ websearch_to_tsquery('english', %(text)s)
    LIMIT %(cand)s
),
fused AS (
    SELECT
        COALESCE(vec.id, txt.id) AS id,
        COALESCE(1.0 / (%(k)s + vec.rank), 0)
      + COALESCE(1.0 / (%(k)s + txt.rank), 0) AS score,
        vec.rank AS vector_rank,
        txt.rank AS text_rank
    FROM vec FULL OUTER JOIN txt ON vec.id = txt.id
)
SELECT c.id, c.source, c.title, c.url, c.heading, c.content,
       f.score, f.vector_rank, f.text_rank
FROM fused f JOIN chunks c ON c.id = f.id
ORDER BY f.score DESC
LIMIT %(top)s;
"""


def search(query: str, vector: list[float], *, top_k: int = 5) -> list[Hit]:
    conn = _connect()
    # Sent as pgvector's own text form and cast, rather than as a Python list.
    # psycopg adapts a list to `double precision[]`, for which no `<=>`
    # operator exists — and the resulting error names the operator rather than
    # the parameter, so it reads like a missing extension.
    literal = "[" + ",".join(f"{v:.7g}" for v in vector) + "]"
    rows = conn.execute(
        SEARCH_SQL,
        {"q": literal, "text": query, "cand": top_k * 4, "k": RRF_K, "top": top_k},
    ).fetchall()
    return [
        Hit(id=r[0], source=r[1], title=r[2], url=r[3], heading=r[4],
            content=r[5], score=float(r[6]), vector_rank=r[7], text_rank=r[8])
        for r in rows
    ]


def main() -> int:
    if "--load" in sys.argv:
        print(f"loading into {dsn().split('password')[0].strip()}")
        load()
        return 0

    if "--query" in sys.argv:
        q = sys.argv[sys.argv.index("--query") + 1]
        from src.retrieval.search import embed_query

        t0 = time.perf_counter()
        hits = search(q, embed_query(q))
        ms = (time.perf_counter() - t0) * 1000
        print(f'"{q}"  ({ms:.0f} ms)\n')
        for i, h in enumerate(hits, 1):
            where = f"{h.title} › {h.heading}" if h.heading else h.title
            print(f"  [{i}] {where}  ({h.source})")
            print(f"      rrf={h.score:.5f}  vec_rank={h.vector_rank}  txt_rank={h.text_rank}")
        return 0

    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
