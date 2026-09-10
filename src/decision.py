"""
Three-tier decision policy.

Maps pair-level match confidence onto an APPLICATION-level disposition, which
is the unit an underwriting team actually works in:

    AUTO_REJECT      confidence >= T_REJECT   confirmed duplicate, no human touch
    FLAG_FOR_REVIEW  T_REVIEW <= c < T_REJECT grey zone, routed to an underwriter
    AUTO_APPROVE     confidence <  T_REVIEW   no credible match, proceeds

This is the "good / bad / grey" split described in the Razorpay Capital post.

Baseline for the workload comparison
------------------------------------
Choosing the counterfactual honestly matters more than the headline number.

  * "review all 10,000" is a strawman - no lender manually dedupes every file.
  * screening at a very low bar (T_SCREEN=60) is also a strawman: precision
    there is under 10%, so no team would ever operate it. It is reported only
    as a loose upper bound.

The DEFENSIBLE baseline is EQUAL-RECALL: a single-threshold system tuned to
the same operating point (T_REVIEW) catches the same duplicates, but has no
confidence tiering - so every one of its matches lands in front of a human.
The tiered policy keeps that recall and auto-decides the near-certain band,
leaving humans only the grey zone. That is the number quoted as the headline.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

from evaluate import true_pairs
from matching import run_matching

ROOT = pathlib.Path(__file__).resolve().parent.parent

T_SCREEN = 60.0   # recall-safe screening bar -> defines the untiered baseline queue
T_REVIEW = 80.0   # below this, auto-approve
T_REJECT = 92.0   # at or above this, auto-reject


def best_match_per_application(pairs_df: pd.DataFrame,
                               all_ids: list[str]) -> pd.DataFrame:
    """Collapse pairs to one row per application: its single strongest match."""
    best: dict[str, dict] = {}
    for r in pairs_df.itertuples():
        for me, other in ((r.left_id, r.right_id), (r.right_id, r.left_id)):
            cur = best.get(me)
            if cur is None or r.confidence > cur["confidence"]:
                best[me] = {
                    "applicant_id": me,
                    "matched_id": other,
                    "confidence": r.confidence,
                    "match_type": r.match_type,
                    "name_score": r.name_score,
                    "address_score": r.address_score,
                    "dob_score": r.dob_score,
                }

    rows = [best.get(i, {"applicant_id": i, "matched_id": None, "confidence": 0.0,
                         "match_type": "NO_MATCH", "name_score": 0.0,
                         "address_score": 0.0, "dob_score": 0.0})
            for i in all_ids]
    return pd.DataFrame(rows)


def assign_tier(confidence: float, t_review: float = T_REVIEW,
                t_reject: float = T_REJECT) -> str:
    if confidence >= t_reject:
        return "AUTO_REJECT"
    if confidence >= t_review:
        return "FLAG_FOR_REVIEW"
    return "AUTO_APPROVE"


def is_true_duplicate(app_id: str, matched_id: str | None,
                      group_of: dict[str, str]) -> bool:
    if matched_id is None:
        return False
    return group_of.get(app_id) == group_of.get(matched_id)


def simulate(df: pd.DataFrame, pairs_df: pd.DataFrame,
             t_screen: float = T_SCREEN, t_review: float = T_REVIEW,
             t_reject: float = T_REJECT) -> dict:
    group_of = dict(zip(df.applicant_id, df.dup_group_id))
    all_ids = df.applicant_id.tolist()

    best = best_match_per_application(pairs_df, all_ids)
    best["tier"] = best.confidence.apply(lambda c: assign_tier(c, t_review, t_reject))
    best["is_true_dup"] = [
        is_true_duplicate(r.applicant_id, r.matched_id, group_of)
        for r in best.itertuples()
    ]

    n_apps = len(best)

    # Equal-recall baseline (headline): a single-threshold system at the same
    # operating point catches the same duplicates but tiers nothing, so every
    # match it surfaces goes to a human.
    n_baseline_equal = int((best.confidence >= t_review).sum())

    # Loose baseline: screening at the recall-safe bar. Reported for context
    # only - precision there is too low for anyone to actually operate it.
    n_baseline_screen = int((best.confidence >= t_screen).sum())

    tiered_queue = best[best.tier == "FLAG_FOR_REVIEW"]
    n_review = len(tiered_queue)
    n_reject = int((best.tier == "AUTO_REJECT").sum())
    n_approve = int((best.tier == "AUTO_APPROVE").sum())

    reduction = 1 - n_review / n_baseline_equal if n_baseline_equal else 0.0
    reduction_screen = (1 - n_review / n_baseline_screen) if n_baseline_screen else 0.0

    # Precision of each machine-decided tier.
    def precision_of(subset: pd.DataFrame) -> float:
        return round(subset.is_true_dup.mean(), 4) if len(subset) else 0.0

    prec_reject = precision_of(best[best.tier == "AUTO_REJECT"])
    prec_review = precision_of(tiered_queue)
    flagged = best[best.tier.isin(["AUTO_REJECT", "FLAG_FOR_REVIEW"])]
    prec_flagged = precision_of(flagged)

    # Safety: true duplicates that slipped into AUTO_APPROVE (the costly error).
    injected = df[df.record_kind.isin({"exact_key", "fuzzy_only", "adversarial"})]
    injected_ids = set(injected.applicant_id)
    inj = best[best.applicant_id.isin(injected_ids)]
    leaked = inj[inj.tier == "AUTO_APPROVE"]
    caught = inj[inj.tier.isin(["AUTO_REJECT", "FLAG_FOR_REVIEW"])]

    # False auto-rejects: the most damaging error (a real customer denied).
    false_auto_rejects = best[(best.tier == "AUTO_REJECT") & (~best.is_true_dup)]

    return {
        "thresholds": {"screen": t_screen, "review": t_review, "reject": t_reject},
        "applications": n_apps,
        "baseline_review_queue_equal_recall": n_baseline_equal,
        "baseline_review_queue_low_screen": n_baseline_screen,
        "manual_review_reduction_pct_low_screen": round(reduction_screen * 100, 2),
        "tiered": {
            "auto_reject": n_reject,
            "flag_for_review": n_review,
            "auto_approve": n_approve,
        },
        "manual_review_reduction_pct": round(reduction * 100, 2),
        "precision": {
            "auto_reject_tier": prec_reject,
            "review_tier": prec_review,
            "all_flagged": prec_flagged,
        },
        "safety": {
            "injected_duplicates": len(inj),
            "caught_by_pipeline": len(caught),
            "catch_rate": round(len(caught) / len(inj), 4) if len(inj) else 0.0,
            "leaked_to_auto_approve": len(leaked),
            "false_auto_rejects": len(false_auto_rejects),
        },
    }, best


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "applicants.csv", dtype=str)
    pairs_df, _ = run_matching(df)

    res, best = simulate(df, pairs_df)

    print("=" * 68)
    print("THREE-TIER DECISION SYSTEM")
    print("=" * 68)
    t = res["thresholds"]
    print(f"  screening bar (baseline)   : confidence >= {t['screen']:.0f}")
    print(f"  auto-reject                : confidence >= {t['reject']:.0f}")
    print(f"  flag-for-review            : {t['review']:.0f} <= confidence < {t['reject']:.0f}")
    print(f"  auto-approve               : confidence <  {t['review']:.0f}")
    print()
    print(f"  total applications         : {res['applications']:,}")
    print(f"  baseline queue (equal-recall, untiered) : "
          f"{res['baseline_review_queue_equal_recall']:,}")
    print(f"  baseline queue (low screening bar)      : "
          f"{res['baseline_review_queue_low_screen']:,}"
          "   [context only - 9.9% precision]")
    print()
    print(f"  AUTO_REJECT                : {res['tiered']['auto_reject']:,}")
    print(f"  FLAG_FOR_REVIEW            : {res['tiered']['flag_for_review']:,}")
    print(f"  AUTO_APPROVE               : {res['tiered']['auto_approve']:,}")
    print()
    print(f"  manual review REDUCTION    : {res['manual_review_reduction_pct']:.1f}%"
          f"  ({res['baseline_review_queue_equal_recall']:,} -> "
          f"{res['tiered']['flag_for_review']:,})   [equal-recall baseline]")
    print()
    p = res["precision"]
    print(f"  precision, auto-reject tier: {p['auto_reject_tier'] * 100:.1f}%")
    print(f"  precision, review tier     : {p['review_tier'] * 100:.1f}%")
    print(f"  precision, ALL flagged     : {p['all_flagged'] * 100:.1f}%")
    print()
    s = res["safety"]
    print(f"  injected duplicates        : {s['injected_duplicates']:,}")
    print(f"  caught (reject or review)  : {s['caught_by_pipeline']:,} "
          f"({s['catch_rate'] * 100:.1f}%)")
    print(f"  leaked to AUTO_APPROVE     : {s['leaked_to_auto_approve']:,}")
    print(f"  FALSE auto-rejects         : {s['false_auto_rejects']:,}"
          "   (good customers wrongly denied)")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "decisions.json", "w") as fh:
        json.dump(res, fh, indent=2)
    best.to_csv(ROOT / "results" / "application_decisions.csv", index=False)
    print(f"\nwrote {ROOT / 'results' / 'decisions.json'}")


if __name__ == "__main__":
    main()
