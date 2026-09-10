"""Unit tests for normalisation, similarity, blocking and tiering."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from decision import assign_tier                                    # noqa: E402
from matching import (address_similarity, build_blocks,             # noqa: E402
                      candidate_pairs, dob_similarity, name_similarity,
                      norm_address, norm_email, norm_phone, run_matching,
                      score_pair, soundex)


# ---------------------------------------------------------------- normalisation

@pytest.mark.parametrize("raw,expected", [
    ("+91 98765 43210", "9876543210"),
    ("098765-43210", "9876543210"),
    ("9876543210", "9876543210"),
])
def test_norm_phone_strips_formatting_and_country_code(raw, expected):
    assert norm_phone(raw) == expected


def test_norm_email_ignores_dots_and_plus_tags():
    assert norm_email("Rajesh.Sharma+loans@Gmail.com") == "rajeshsharma@gmail.com"


def test_norm_address_expands_abbreviations():
    assert norm_address("12 MG Rd, Sec 45") == norm_address("12 MG Road, Sector 45")


def test_soundex_groups_phonetic_spellings():
    assert soundex("Sharma") == soundex("Sarma")
    assert soundex("Chauhan") == soundex("Chouhan")
    assert soundex("Sharma") != soundex("Gupta")


# ------------------------------------------------------------------ similarity

def test_name_similarity_handles_reordering():
    assert name_similarity("Rajesh Kumar Sharma", "Sharma Rajesh Kumar") >= 95


def test_name_similarity_handles_initials():
    assert name_similarity("Rajesh Kumar Sharma", "R. K. Sharma") >= 80


def test_name_similarity_tolerates_single_typo():
    assert name_similarity("Priya Menon", "Priya Menno") >= 80


def test_name_similarity_separates_different_people():
    assert name_similarity("Rajesh Sharma", "Kavita Iyer") < 50


def test_dob_similarity_scores_transposition_above_random():
    assert dob_similarity("1988-04-11", "1988-04-11") == 100
    assert dob_similarity("1988-04-11", "1988-11-04") == 90
    assert dob_similarity("1988-04-11", "1988-04-12") == 75
    assert dob_similarity("1988-04-11", "1991-07-23") == 0


def test_address_similarity_survives_abbreviation_and_reorder():
    a = "Flat 402, 12 MG Road, Indiranagar"
    b = "12 MG Rd, Indiranagar, Flat 402"
    assert address_similarity(a, b) >= 90


# -------------------------------------------------------------------- blocking

def _row(**kw):
    base = dict(applicant_id="APP1", full_name="Rajesh Kumar Sharma",
                dob="1988-04-11", phone="9876543210", pan="ABCDE1234F",
                email="rajesh@gmail.com", address="12 MG Road, Indiranagar",
                city="Bengaluru", state="Karnataka", pincode="560038")
    base.update(kw)
    return base


def test_blocking_places_exact_phone_matches_in_one_block():
    df = pd.DataFrame([
        _row(applicant_id="APP1"),
        _row(applicant_id="APP2", pan="ZZZZZ9999Z", full_name="R. K. Sharma"),
    ])
    blocks = build_blocks(df)
    assert len(blocks["phone:9876543210"]) == 2


def test_candidate_pairs_skips_oversized_blocks():
    df = pd.DataFrame([_row(applicant_id=f"APP{i}") for i in range(10)])
    blocks = build_blocks(df)
    _, stats = candidate_pairs(blocks, max_block_size=5)
    assert stats["skipped_oversized_blocks"] > 0


# ----------------------------------------------------------------- pair scoring

def test_exact_pan_dominates_score():
    left = pd.Series(_row(applicant_id="APP1"))
    right = pd.Series(_row(applicant_id="APP2", full_name="Totally Different Person",
                           address="99 Nowhere Lane", phone="9000000000",
                           email="x@y.com"))
    ps = score_pair(left, right)
    assert ps.match_type == "EXACT_PAN"
    assert ps.confidence == 100.0


def test_exact_phone_recognised_when_pan_differs():
    left = pd.Series(_row(applicant_id="APP1"))
    right = pd.Series(_row(applicant_id="APP2", pan="ZZZZZ9999Z",
                           email="other@x.com"))
    ps = score_pair(left, right)
    assert ps.match_type == "EXACT_PHONE"
    assert ps.phone_exact and not ps.pan_exact


def test_unrelated_records_score_low():
    left = pd.Series(_row(applicant_id="APP1"))
    right = pd.Series(_row(applicant_id="APP2", full_name="Kavita Iyer",
                           dob="1975-01-02", phone="9111111111",
                           pan="ZZZZZ9999Z", email="k@x.com",
                           address="7 Anna Salai, Adyar", city="Chennai",
                           state="Tamil Nadu", pincode="600020"))
    assert score_pair(left, right).confidence < 50


# ---------------------------------------------------------------------- tiering

@pytest.mark.parametrize("conf,tier", [
    (100.0, "AUTO_REJECT"), (92.0, "AUTO_REJECT"),
    (91.9, "FLAG_FOR_REVIEW"), (80.0, "FLAG_FOR_REVIEW"),
    (79.9, "AUTO_APPROVE"), (0.0, "AUTO_APPROVE"),
])
def test_tier_boundaries(conf, tier):
    assert assign_tier(conf) == tier


# ------------------------------------------------------------ end-to-end smoke

def test_pipeline_finds_a_planted_duplicate():
    df = pd.DataFrame([
        _row(applicant_id="APP1"),
        # same person: light typo, abbreviated address, new PAN, same phone
        _row(applicant_id="APP2", full_name="Rajesh Kumar Sarma",
             pan="ZZZZZ9999Z", email="rajesh.k@gmail.com",
             address="12 MG Rd, Indiranagar"),
        _row(applicant_id="APP3", full_name="Kavita Iyer", dob="1975-01-02",
             phone="9111111111", pan="QQQQQ1111Q", email="k@x.com",
             address="7 Anna Salai, Adyar", city="Chennai",
             state="Tamil Nadu", pincode="600020"),
    ])
    pairs, _ = run_matching(df, min_confidence=50.0)
    matched = {frozenset((r.left_id, r.right_id)) for r in pairs.itertuples()
               if r.confidence >= 80}
    assert frozenset(("APP1", "APP2")) in matched
    assert frozenset(("APP1", "APP3")) not in matched
