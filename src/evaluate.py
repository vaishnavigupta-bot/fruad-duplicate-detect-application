"""
Evaluation harness.

Answers three questions, in order:

  1. BLOCKING RECALL   Does candidate generation keep the true duplicate pairs?
                       (A pair dropped here can never be recovered downstream.)
  2. THRESHOLD SWEEP   Recall / precision / F1 across similarity thresholds,
                       broken out by injected-duplicate difficulty tier.
  3. TIER SIMULATION   How the three-tier decision policy reshapes the manual
                       review queue.

Ground truth: two records are duplicates iff they share `dup_group_id`.
"""

from __future__ import annotations

import json
import pathlib
from itertools import combinations

import pandas as pd

from matching import run_matching

ROOT = pathlib.Path(__file__).resolve().parent.parent
DUP_KINDS = {"exact_key", "fuzzy_only", "adversarial"}


def true_pairs(df: pd.DataFrame) -> set[frozenset[str]]:
    """All unordered id-pairs that share a dup_group_id."""
    out = set()
    for _, grp in df.groupby("dup_group_id"):
        if len(grp) > 1:
            for a, b in combinations(sorted(grp.applicant_id), 2):
                out.add(frozenset((a, b)))
    return out


def pair_key(row) -> frozenset[str]:
    return frozenset((row.left_id, row.right_id))


def evaluate_threshold(pairs_df: pd.DataFrame, truth: set[frozenset[str]],
                       threshold: float) -> dict:
    """Pair-level precision/recall at a confidence cutoff."""
    flagged = pairs_df[pairs_df.confidence >= threshold]
    predicted = {frozenset((r.left_id, r.right_id)) for r in flagged.itertuples()}

    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "predicted_pairs": len(predicted),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def recall_by_kind(pairs_df: pd.DataFrame, df: pd.DataFrame,
                   threshold: float) -> dict[str, dict]:
    """
    Record-level recall on injected duplicates, split by difficulty tier.

    A duplicate record counts as RECALLED if it is linked, at or above the
    threshold, to any other record in its true group.
    """
    flagged = pairs_df[pairs_df.confidence >= threshold]
    linked: dict[str, set[str]] = {}
    for r in flagged.itertuples():
        linked.setdefault(r.left_id, set()).add(r.right_id)
        linked.setdefault(r.right_id, set()).add(r.left_id)

    group_of = dict(zip(df.applicant_id, df.dup_group_id))
    out: dict[str, dict] = {}

    for kind in sorted(DUP_KINDS):
        subset = df[df.record_kind == kind]
        hit = 0
        for rec_id in subset.applicant_id:
            partners = linked.get(rec_id, set())
            if any(group_of.get(p) == group_of[rec_id] for p in partners):
                hit += 1
        total = len(subset)
        out[kind] = {
            "records": total,
            "recalled": hit,
            "recall": round(hit / total, 4) if total else 0.0,
        }

    total = sum(v["records"] for v in out.values())
    recalled = sum(v["recalled"] for v in out.values())
    out["ALL_INJECTED"] = {
        "records": total,
        "recalled": recalled,
        "recall": round(recalled / total, 4) if total else 0.0,
    }
    return out


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "applicants.csv", dtype=str)
    pairs_df, stats = run_matching(df)
    truth = true_pairs(df)

    # ---- 1. blocking recall -------------------------------------------------
    candidates = {frozenset((r.left_id, r.right_id)) for r in pairs_df.itertuples()}
    blocking_recall = len(candidates & truth) / len(truth)

    print("=" * 68)
    print("1. BLOCKING / CANDIDATE GENERATION")
    print("=" * 68)
    print(f"  records                  : {stats['n_records']:,}")
    print(f"  all possible pairs       : {stats['all_possible_pairs']:,}")
    print(f"  candidate pairs          : {stats['candidate_pairs']:,}")
    print(f"  search-space reduction   : {stats['reduction_ratio'] * 100:.4f}%")
    print(f"  true duplicate pairs     : {len(truth):,}")
    print(f"  captured by blocking     : {len(candidates & truth):,} "
          f"({blocking_recall * 100:.2f}% blocking recall)")

    # ---- 2. threshold sweep -------------------------------------------------
    print()
    print("=" * 68)
    print("2. THRESHOLD SWEEP (pair-level)")
    print("=" * 68)
    print(f"  {'thr':>5} {'flagged':>9} {'TP':>6} {'FP':>7} {'FN':>5} "
          f"{'prec':>7} {'recall':>7} {'F1':>7}")

    sweep = []
    for thr in range(60, 100, 2):
        m = evaluate_threshold(pairs_df, truth, float(thr))
        sweep.append(m)
        print(f"  {thr:>5} {m['predicted_pairs']:>9,} {m['true_positives']:>6,} "
              f"{m['false_positives']:>7,} {m['false_negatives']:>5,} "
              f"{m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f}")

    best_f1 = max(sweep, key=lambda m: m["f1"])
    print(f"\n  best F1 at threshold {best_f1['threshold']:.0f}: "
          f"P={best_f1['precision']:.3f} R={best_f1['recall']:.3f} "
          f"F1={best_f1['f1']:.3f}")

    # ---- 3. recall by difficulty tier --------------------------------------
    print()
    print("=" * 68)
    print("3. RECORD-LEVEL RECALL BY INJECTED-DUPLICATE DIFFICULTY")
    print("=" * 68)
    by_kind = {}
    for thr in (80.0, 85.0, best_f1["threshold"], 90.0):
        by_kind[thr] = recall_by_kind(pairs_df, df, thr)

    kinds = ["exact_key", "fuzzy_only", "adversarial", "ALL_INJECTED"]
    header = "  " + f"{'threshold':>10}" + "".join(f"{k:>16}" for k in kinds)
    print(header)
    for thr in sorted(by_kind):
        cells = "".join(
            f"{by_kind[thr][k]['recall'] * 100:>15.1f}%" for k in kinds)
        print(f"  {thr:>10.0f}{cells}")

    results = {
        "blocking": {**stats, "blocking_recall": round(blocking_recall, 4),
                     "true_pairs": len(truth)},
        "sweep": sweep,
        "best_f1": best_f1,
        "recall_by_kind": {str(k): v for k, v in by_kind.items()},
    }
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "evaluation.json", "w") as fh:
        json.dump(results, fh, indent=2)
    pairs_df.to_csv(ROOT / "results" / "scored_pairs.csv", index=False)
    print(f"\nwrote {ROOT / 'results' / 'evaluation.json'}")


if __name__ == "__main__":
    main()
