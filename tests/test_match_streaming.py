"""
Streaming (per-applicant) screening must agree with the batch pipeline.

There is no live Elasticsearch node in CI (see `search_index.py`'s docstring -
the search benchmark is excluded from CI by design). So these tests exercise
`decide_from_candidates`, the ES-independent half of `match_streaming.py`,
directly: it takes a plain list of candidate dicts (in production, an ES
hit list; here, a hand-built one) and must reach the same tier as
`run_matching` + `assign_tier` reach for the exact same pairs.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from decision import assign_tier                                    # noqa: E402
from match_streaming import decide_from_candidates                  # noqa: E402
from matching import run_matching                                   # noqa: E402


def _row(**kw):
    base = dict(applicant_id="APP1", full_name="Rajesh Kumar Sharma",
               address="Flat 402, 12 MG Road, Indiranagar", dob="1988-04-11",
               phone="9876543210", pan="ABCDE1234F", email="rajesh.sharma@gmail.com",
               city="Bengaluru", state="Karnataka", pincode="560038")
    base.update(kw)
    return base


APPLICANTS = [
    _row(applicant_id="APP1"),
    # same person: light typo, abbreviated address, new PAN, same phone
    _row(applicant_id="APP2", full_name="Rajesh Kumar Sarma",
         pan="ZZZZZ9999Z", email="rajesh.k@gmail.com",
         address="12 MG Rd, Indiranagar"),
    # unrelated applicant, shares nothing
    _row(applicant_id="APP3", full_name="Kavita Iyer", dob="1975-01-02",
         phone="9111111111", pan="QQQQQ1111Q", email="k@x.com",
         address="7 Anna Salai, Adyar", city="Chennai",
         state="Tamil Nadu", pincode="600020"),
    # exact PAN reuse of APP1 under a different name - the near-certain case
    _row(applicant_id="APP4", full_name="R K Sharma", pan="ABCDE1234F",
         phone="9000000000", email="rksharma@yahoo.com"),
]


def test_streaming_decision_matches_batch_tier_for_each_applicant():
    df = pd.DataFrame(APPLICANTS)
    pairs, _ = run_matching(df, min_confidence=0.0)

    for applicant in APPLICANTS:
        me = applicant["applicant_id"]
        # The "ES hit list" for this applicant: everyone else, exactly the
        # candidate universe the batch pipeline itself scores it against.
        candidates = [a for a in APPLICANTS if a["applicant_id"] != me]
        streamed = decide_from_candidates(applicant, candidates)

        mine = pairs[(pairs.left_id == me) | (pairs.right_id == me)]
        batch_best = mine["confidence"].max() if not mine.empty else None
        batch_tier = assign_tier(batch_best) if batch_best is not None else "AUTO_APPROVE"

        assert streamed["disposition"] == batch_tier, me
        if batch_best is not None:
            assert streamed["confidence"] == batch_best


def test_streaming_decision_ignores_self_hit():
    # Candidate list containing the applicant's own record (as an ES top-k
    # result naturally would, before the must_not filter) must not self-match.
    applicant = APPLICANTS[0]
    result = decide_from_candidates(applicant, APPLICANTS)
    assert result["matched_id"] != applicant["applicant_id"]


def test_streaming_decision_no_candidates_auto_approves():
    result = decide_from_candidates(APPLICANTS[0], [])
    assert result["disposition"] == "AUTO_APPROVE"
    assert result["matched_id"] is None
