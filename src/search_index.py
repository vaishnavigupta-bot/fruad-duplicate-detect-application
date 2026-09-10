"""
Elasticsearch-backed applicant search + latency benchmark.

Why a search engine at all
--------------------------
The offline pipeline (`matching.py`) dedupes a whole batch against itself. A
live underwriting system has the opposite shape: ONE new application arrives
and must be screened against the entire existing book in real time. That is a
per-query latency problem, not a batch problem - which is exactly why Razorpay
Capital chose Elasticsearch over Postgres/Hive/DynamoDB for this component.

Index design
------------
`full_name` and `address` are indexed with custom analysers (lowercasing,
ASCII folding, address-abbreviation synonyms) and also kept as `.keyword`
subfields. The dedupe query is a single `bool` with:

    should: exact term on phone / pan / email      (high boost, deterministic)
    should: fuzzy match on name                    (AUTO edit distance)
    should: fuzzy match on address                 (analysed, order-tolerant)
    should: term on dob

so one round trip returns a ranked candidate list scored by BM25 plus boosts.
Latency is measured over that single call.

The client is `elasticsearch` (8.x). The same index body and query DSL run
unchanged against OpenSearch via `opensearch-py`; see `README.md`.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import time

import pandas as pd
from elasticsearch import Elasticsearch, helpers

ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX = "loan_applicants"
ES_URL = "http://localhost:9200"

SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "analysis": {
        "filter": {
            # Canonicalises free-text address variance at index and query time.
            "address_synonyms": {
                "type": "synonym",
                "synonyms": [
                    "rd, road", "st, street", "ave, avenue",
                    "apt, apartment", "bldg, building", "flr, floor",
                    "nr, near", "opp, opposite", "sec, sector",
                ],
            },
        },
        "analyzer": {
            "identity_analyzer": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "asciifolding"],
            },
            "address_analyzer": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "asciifolding", "address_synonyms"],
            },
        },
    },
}

MAPPINGS = {
    "properties": {
        "applicant_id": {"type": "keyword"},
        "dup_group_id": {"type": "keyword"},
        "record_kind": {"type": "keyword"},
        "full_name": {
            "type": "text",
            "analyzer": "identity_analyzer",
            "fields": {"keyword": {"type": "keyword"}},
        },
        "address": {
            "type": "text",
            "analyzer": "address_analyzer",
            "fields": {"keyword": {"type": "keyword"}},
        },
        "phone": {"type": "keyword"},
        "pan": {"type": "keyword"},
        "email": {"type": "keyword"},
        "dob": {"type": "keyword"},
        "city": {"type": "keyword"},
        "state": {"type": "keyword"},
        "pincode": {"type": "keyword"},
        "loan_amount": {"type": "long"},
        "loan_purpose": {"type": "keyword"},
        "applied_on": {"type": "keyword"},
    }
}


def client() -> Elasticsearch:
    return Elasticsearch(ES_URL, request_timeout=60)


def build_index(es: Elasticsearch, df: pd.DataFrame) -> float:
    """(Re)create the index and bulk-load every applicant. Returns seconds taken."""
    if es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)
    es.indices.create(index=INDEX, settings=SETTINGS, mappings=MAPPINGS)

    def actions():
        for row in df.to_dict("records"):
            row = dict(row)
            row["loan_amount"] = int(row["loan_amount"])
            yield {"_index": INDEX, "_id": row["applicant_id"], "_source": row}

    t0 = time.perf_counter()
    helpers.bulk(es, actions(), chunk_size=1000, request_timeout=180)
    es.indices.refresh(index=INDEX)
    return time.perf_counter() - t0


def dedupe_query(applicant: dict) -> dict:
    """
    The single-round-trip dedupe query for one incoming application.

    Deterministic identifiers carry large boosts so an exact PAN/phone hit
    always outranks a merely similar name.
    """
    return {
        "bool": {
            "must_not": [{"term": {"applicant_id": applicant["applicant_id"]}}],
            "should": [
                {"term": {"pan": {"value": applicant["pan"], "boost": 12.0}}},
                {"term": {"phone": {"value": applicant["phone"], "boost": 10.0}}},
                {"term": {"email": {"value": applicant["email"], "boost": 8.0}}},
                {"term": {"dob": {"value": applicant["dob"], "boost": 3.0}}},
                {"match": {"full_name": {
                    "query": applicant["full_name"],
                    "fuzziness": "AUTO",
                    "operator": "or",
                    "boost": 5.0,
                }}},
                {"match": {"address": {
                    "query": applicant["address"],
                    "fuzziness": "AUTO",
                    "operator": "or",
                    "boost": 2.0,
                }}},
            ],
            "minimum_should_match": 1,
        }
    }


def benchmark(es: Elasticsearch, df: pd.DataFrame, n_queries: int = 1000,
              warmup: int = 50, size: int = 10, seed: int = 7) -> dict:
    """Measure end-to-end per-query latency over a random sample of applicants."""
    sample = df.sample(n=n_queries + warmup, random_state=seed).to_dict("records")

    for rec in sample[:warmup]:
        es.search(index=INDEX, query=dedupe_query(rec), size=size)

    latencies_ms: list[float] = []
    took_ms: list[int] = []
    for rec in sample[warmup:]:
        q = dedupe_query(rec)
        t0 = time.perf_counter()
        resp = es.search(index=INDEX, query=q, size=size)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        took_ms.append(resp["took"])

    latencies_ms.sort()

    def pct(p: float) -> float:
        idx = min(int(len(latencies_ms) * p), len(latencies_ms) - 1)
        return round(latencies_ms[idx], 2)

    return {
        "index": INDEX,
        "indexed_documents": int(es.count(index=INDEX)["count"]),
        "queries": len(latencies_ms),
        "top_k": size,
        "client_side_latency_ms": {
            "mean": round(statistics.mean(latencies_ms), 2),
            "median_p50": pct(0.50),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "min": round(latencies_ms[0], 2),
            "max": round(latencies_ms[-1], 2),
        },
        "server_side_took_ms": {
            "mean": round(statistics.mean(took_ms), 2),
            "max": max(took_ms),
        },
        "throughput_qps_single_client": round(1000 / statistics.mean(latencies_ms), 1),
    }


def search_recall(es: Elasticsearch, df: pd.DataFrame, top_k: int = 5) -> dict:
    """
    Retrieval quality: for each injected duplicate, is its true partner in the
    top-k hits? Latency only matters if the right record actually comes back.
    """
    injected = df[df.record_kind.isin({"exact_key", "fuzzy_only", "adversarial"})]
    group_of = dict(zip(df.applicant_id, df.dup_group_id))

    by_kind: dict[str, dict] = {}
    for rec in injected.to_dict("records"):
        bucket = by_kind.setdefault(rec["record_kind"], {"n": 0, "hit": 0})
        bucket["n"] += 1
        resp = es.search(index=INDEX, query=dedupe_query(rec), size=top_k)
        hits = [h["_source"]["applicant_id"] for h in resp["hits"]["hits"]]
        if any(group_of.get(h) == group_of[rec["applicant_id"]] for h in hits):
            bucket["hit"] += 1

    for v in by_kind.values():
        v["recall_at_k"] = round(v["hit"] / v["n"], 4)

    total_n = sum(v["n"] for v in by_kind.values())
    total_hit = sum(v["hit"] for v in by_kind.values())
    by_kind["ALL"] = {"n": total_n, "hit": total_hit,
                      "recall_at_k": round(total_hit / total_n, 4)}
    return {"top_k": top_k, "by_kind": by_kind}


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "applicants.csv", dtype=str)
    es = client()

    info = es.info()
    print(f"connected to Elasticsearch {info['version']['number']} "
          f"(lucene {info['version']['lucene_version']})\n")

    print(f"indexing {len(df):,} applicants ...")
    secs = build_index(es, df)
    print(f"  indexed in {secs:.2f}s ({len(df) / secs:,.0f} docs/sec)\n")

    print("benchmarking single-application dedupe queries ...")
    bench = benchmark(es, df)
    lat = bench["client_side_latency_ms"]
    print(f"  documents in index : {bench['indexed_documents']:,}")
    print(f"  queries measured   : {bench['queries']:,}")
    print(f"  mean latency       : {lat['mean']} ms")
    print(f"  p50 / p90 / p95    : {lat['median_p50']} / {lat['p90']} / {lat['p95']} ms")
    print(f"  p99 / max          : {lat['p99']} / {lat['max']} ms")
    print(f"  server-side 'took'  : {bench['server_side_took_ms']['mean']} ms mean")
    print(f"  throughput         : {bench['throughput_qps_single_client']:,} q/s "
          "(single client, sequential)\n")

    print("measuring retrieval recall@5 ...")
    rec = search_recall(es, df)
    for kind, v in rec["by_kind"].items():
        print(f"  {kind:<16} {v['hit']:>4}/{v['n']:<4} = {v['recall_at_k'] * 100:.1f}%")

    out = {
        "engine": f"elasticsearch {info['version']['number']}",
        "indexing_seconds": round(secs, 2),
        "benchmark": bench,
        "retrieval": rec,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "search_benchmark.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {ROOT / 'results' / 'search_benchmark.json'}")


if __name__ == "__main__":
    main()
