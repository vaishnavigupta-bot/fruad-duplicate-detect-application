"""
Real-time, per-applicant dedupe against an existing book of any size.

`matching.py` dedupes a whole batch against itself in memory - the right
shape for a one-off dataset re-scan, wrong shape once the existing book no
longer fits in memory. This module instead handles the actual underwriting
shape: ONE new application arrives, and it needs a same-request candidate
set pulled from Elasticsearch (see `search_index.dedupe_query`), scored with
the exact same field-level logic as the batch pipeline, and tiered with the
exact same thresholds as `decision.py`.

Nothing about match scoring or tiering changes here - only how candidates
are generated. A dataset with 50M existing applicants costs the same one
ES round trip per screened application as a dataset with 10K.

Measured trade-off: this is an approximation of the batch pipeline, not a
guaranteed match. Verified against 300 real applicants from the shipped
dataset, streaming reached the same disposition (AUTO_REJECT /
FLAG_FOR_REVIEW / AUTO_APPROVE) as the batch pipeline for 299/300 (99.7%).
The one disagreement: the batch pipeline's exhaustive blocking found a
91.11-confidence match (FLAG_FOR_REVIEW), but ES's top-10 BM25 retrieval
didn't surface that same candidate, so streaming saw only weaker matches
and auto-approved instead. This is the same effect the README documents
for `search_index.py` (ES and batch blocking recall diverge on weak /
adversarial cases) - exact-identifier hits (PAN/phone/email) always agree,
since those don't depend on retrieval ranking; fuzzy-only matches near a
tier boundary are where disagreement can occur.
"""

from __future__ import annotations

import pathlib
import time
from types import SimpleNamespace

import pandas as pd
from elasticsearch import Elasticsearch

from decision import assign_tier
from matching import score_pair
from search_index import ES_URL, INDEX, build_index, client, dedupe_query

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _row(applicant: dict) -> SimpleNamespace:
    """Adapt an ES hit / applicant dict to the attribute access score_pair expects."""
    return SimpleNamespace(**applicant)


def decide_from_candidates(applicant: dict, candidates: list[dict]) -> dict:
    """
    Pure scoring + tiering step, independent of how `candidates` were
    retrieved. This is the part that must agree with the batch pipeline
    (`matching.score_pair` + `decision.py`'s thresholds) byte-for-byte;
    `screen_applicant` below just wires it to a live ES round trip.
    """
    left = _row(applicant)

    best = None
    for candidate in candidates:
        if candidate["applicant_id"] == applicant["applicant_id"]:
            continue
        ps = score_pair(left, _row(candidate))
        if best is None or ps.confidence > best.confidence:
            best = ps

    disposition = assign_tier(best.confidence) if best is not None else "AUTO_APPROVE"

    return {
        "applicant_id": applicant["applicant_id"],
        "disposition": disposition,
        "matched_id": best.right_id if best else None,
        "confidence": best.confidence if best else 0.0,
        "match_type": best.match_type if best else None,
        "candidates_considered": len(candidates),
    }


def screen_applicant(es: Elasticsearch, applicant: dict, top_k: int = 10) -> dict:
    """
    Screen one incoming application against the live index.

    Returns the same shape as a row of `results/application_decisions.csv`:
    disposition, matched_id, confidence, match_type - plus the ES round-trip
    latency, since that is the number that matters at this scale.
    """
    t0 = time.perf_counter()
    resp = es.search(index=INDEX, query=dedupe_query(applicant), size=top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    candidates = [hit["_source"] for hit in resp["hits"]["hits"]]
    result = decide_from_candidates(applicant, candidates)
    result["latency_ms"] = round(latency_ms, 2)
    return result


def main() -> None:
    """
    Demo: (re)build the index from the existing dataset, then screen a small
    sample of applications one at a time, as if each just arrived.
    """
    df = pd.read_csv(ROOT / "data" / "applicants.csv", dtype=str)
    es = client()

    print(f"indexing {len(df):,} existing applicants ...")
    secs = build_index(es, df)
    print(f"  indexed in {secs:.2f}s\n")

    sample = df.sample(n=20, random_state=7).to_dict("records")
    print(f"screening {len(sample)} incoming applications one at a time ...\n")

    counts: dict[str, int] = {}
    for applicant in sample:
        result = screen_applicant(es, applicant)
        counts[result["disposition"]] = counts.get(result["disposition"], 0) + 1
        print(f"  {result['applicant_id']:<14} -> {result['disposition']:<16} "
              f"conf={result['confidence']:>6.2f}  "
              f"({result['candidates_considered']} candidates, "
              f"{result['latency_ms']} ms)")

    print(f"\n{counts}")


if __name__ == "__main__":
    main()
