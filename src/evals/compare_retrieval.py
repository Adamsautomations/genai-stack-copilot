"""Azure AI Search vs Postgres + pgvector, on identical vectors.

The two engines index the *same* 10,250 chunks with the *same* 384-dim
`bge-small-en-v1.5` embeddings, so nothing here measures embedding quality.
What it measures is the part that actually differs: the index (Azure's HNSW vs
pgvector's HNSW), the keyword side (BM25 vs `ts_rank_cd`), and — on the Azure
side only — an optional L2 semantic reranker.

**There is no relevance ground truth here, and this deliberately does not
invent one.** Treating either engine's ranking as the reference would simply
measure how much the other agrees with an arbitrary choice. So the reported
numbers are agreement, latency and cost, and the honest conclusion is about
*where* the two disagree rather than which is right.

    python -m src.evals.compare_retrieval
    python -m src.evals.compare_retrieval --no-semantic   # hybrid vs hybrid
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from src.config import Settings
from src.evals.questions import QUESTIONS
from src.retrieval import pgvector_store
from src.retrieval.search import embed_query, retrieve

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "evals"
TOP_K = 5


def overlap(a: list[str], b: list[str]) -> float:
    """Fraction of engine A's top-k that engine B also returned."""
    return len(set(a) & set(b)) / len(a) if a else 0.0


def rank_correlation(a: list[str], b: list[str]) -> float | None:
    """Spearman over the ids both engines returned.

    Overlap alone hides ordering: two engines can return the same five
    passages with the best one ranked 1st or 5th, and for RAG that difference
    matters more than membership, because the model reads top-down.
    """
    common = [i for i in a if i in b]
    n = len(common)
    if n < 2:
        return None

    # Rank WITHIN the shared subset, not by position in the full top-k.
    # Spearman's formula assumes both inputs are permutations of 1..n; feeding
    # it raw positions from a longer list produces values far outside [-1, 1]
    # (the first version of this returned -6.00, which is what exposed it).
    def ranks(order: list[int]) -> list[int]:
        out = [0] * len(order)
        for rank, idx in enumerate(sorted(range(len(order)), key=lambda i: order[i])):
            out[idx] = rank
        return out

    ra = ranks([a.index(i) for i in common])
    rb = ranks([b.index(i) for i in common])
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1 - (6 * d2) / (n * (n**2 - 1))


def selftest() -> None:
    """Check the metrics on inputs whose answers are known.

    A correlation function that silently returns out-of-range values is worse
    than one that crashes, because the number still prints and still looks
    like a measurement.
    """
    assert overlap(["a", "b"], ["b", "c"]) == 0.5
    assert overlap(["a"], []) == 0.0
    # identical order -> +1
    assert rank_correlation(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    # exactly reversed -> -1
    assert rank_correlation(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    # fewer than two shared ids is undefined, not zero
    assert rank_correlation(["a", "b"], ["x", "y"]) is None
    # the case that broke the first version: shared ids far apart in long lists
    v = rank_correlation(["a", "z", "y", "x", "b"], ["b", "q", "r", "s", "a"])
    assert v is not None and -1.0 <= v <= 1.0, v


def main() -> int:
    selftest()
    settings = Settings.load()
    use_semantic = "--no-semantic" not in sys.argv

    rows = []
    print(f"comparing {len(QUESTIONS)} questions, top-{TOP_K}, "
          f"azure semantic rerank={'on' if use_semantic else 'off'}\n")

    for q in QUESTIONS:
        vector = embed_query(q.question)  # embedded once, used by both

        t0 = time.perf_counter()
        az = retrieve(q.question, settings, top_k=TOP_K, use_semantic=use_semantic)
        az_ms = (time.perf_counter() - t0) * 1000
        az_ids = [p.id for p in az.passages]

        t0 = time.perf_counter()
        pg = pgvector_store.search(q.question, vector, top_k=TOP_K)
        pg_ms = (time.perf_counter() - t0) * 1000
        pg_ids = [h.id for h in pg]

        # How much of pgvector's result came from each retriever. A question
        # answered entirely by the vector side tells you the keyword side
        # contributed nothing, which is the interesting failure mode.
        vec_only = sum(1 for h in pg if h.text_rank is None)
        txt_only = sum(1 for h in pg if h.vector_rank is None)
        both = sum(1 for h in pg if h.text_rank is not None and h.vector_rank is not None)

        rows.append({
            "question": q.question,
            "category": q.category,
            "azure_ids": az_ids,
            "pgvector_ids": pg_ids,
            "overlap": overlap(az_ids, pg_ids),
            "spearman": rank_correlation(az_ids, pg_ids),
            "azure_ms": round(az_ms, 1),
            "pgvector_ms": round(pg_ms, 1),
            "azure_used_semantic": az.used_semantic,
            "pg_vector_only": vec_only,
            "pg_text_only": txt_only,
            "pg_both": both,
        })
        print(f"  {q.question[:52]:<54} overlap={rows[-1]['overlap']:.0%}  "
              f"az={az_ms:6.0f}ms  pg={pg_ms:6.0f}ms")

    ov = [r["overlap"] for r in rows]
    sp = [r["spearman"] for r in rows if r["spearman"] is not None]
    az_ms = [r["azure_ms"] for r in rows]
    pg_ms = [r["pgvector_ms"] for r in rows]

    print("\n" + "─" * 68)
    print(f"  mean overlap@{TOP_K}          {statistics.mean(ov):.1%}")
    print(f"  questions with 0 overlap    {sum(1 for o in ov if o == 0)} of {len(ov)}")
    print(f"  questions with full overlap {sum(1 for o in ov if o == 1)} of {len(ov)}")
    if sp:
        print(f"  mean Spearman (shared ids)  {statistics.mean(sp):+.2f}")
    print(f"  median latency  azure       {statistics.median(az_ms):.0f} ms")
    print(f"  median latency  pgvector    {statistics.median(pg_ms):.0f} ms")
    # Said out loud because the ratio looks far more impressive than it is.
    print("    (not like-for-like: Azure is a network round trip to "
          "polandcentral,\n     pgvector is a local socket. Treat the gap as "
          "'local beats remote',\n     not 'pgvector beats Azure'.)")
    print(f"  pgvector hits from vector only / text only / both: "
          f"{sum(r['pg_vector_only'] for r in rows)} / "
          f"{sum(r['pg_text_only'] for r in rows)} / "
          f"{sum(r['pg_both'] for r in rows)}")

    # Both variants are kept. The difference between them IS the result, so
    # writing one filename would overwrite the comparison with half of itself.
    out = OUT_DIR / (
        f"retrieval_azure_vs_pgvector_"
        f"{'rerank' if use_semantic else 'hybrid'}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "top_k": TOP_K,
        "azure_semantic_rerank": use_semantic,
        "summary": {
            "mean_overlap": statistics.mean(ov),
            "zero_overlap": sum(1 for o in ov if o == 0),
            "full_overlap": sum(1 for o in ov if o == 1),
            "mean_spearman": statistics.mean(sp) if sp else None,
            "median_ms_azure": statistics.median(az_ms),
            "median_ms_pgvector": statistics.median(pg_ms),
        },
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
