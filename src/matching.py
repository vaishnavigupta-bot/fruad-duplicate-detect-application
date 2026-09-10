"""
Duplicate / fraud applicant matching engine.

Two-stage design, mirroring the Razorpay Capital dedupe architecture:

  Stage 1  BLOCKING      cheap deterministic keys reduce ~50M possible pairs to
                         a few tens of thousands of candidates.
  Stage 2  SCORING       each candidate pair gets an exact-identifier check
                         (phone, PAN, email) and a fuzzy identity comparison
                         (name, address, DOB) via RapidFuzz.

The output is a match-confidence score in [0, 100] per pair, plus the
`match_type` that drove it. Decision tiering lives in `decision.py`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz

# A blocking key shared by more than this many records is uninformative
# (e.g. a whole pincode) and is skipped to keep candidate generation linear.
MAX_BLOCK_SIZE = 60

ADDRESS_CANONICAL = {
    "rd": "road", "st": "street", "ave": "avenue", "apt": "apartment",
    "bldg": "building", "flr": "floor", "nr": "near", "opp": "opposite",
    "sec": "sector", "e": "east", "w": "west", "n": "north", "s": "south",
    "marg.": "marg", "no": "number", "flat": "flat",
}

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_MULTISPACE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def norm_text(value: str) -> str:
    value = _NON_ALNUM.sub(" ", str(value).lower())
    return _MULTISPACE.sub(" ", value).strip()


def norm_name(name: str) -> str:
    return norm_text(name)


def norm_address(address: str) -> str:
    tokens = norm_text(address).split()
    return " ".join(ADDRESS_CANONICAL.get(t, t) for t in tokens)


def norm_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", str(phone))
    return digits[-10:] if len(digits) >= 10 else digits


def norm_email(email: str) -> str:
    """Strip dots/underscores and +tags from the local part (gmail-style)."""
    email = str(email).lower().strip()
    if "@" not in email:
        return email
    local, domain = email.split("@", 1)
    local = local.split("+")[0].replace(".", "").replace("_", "")
    return f"{local}@{domain}"


def soundex(token: str) -> str:
    """Classic Soundex — cheap phonetic key for surname blocking."""
    token = re.sub(r"[^a-z]", "", token.lower())
    if not token:
        return ""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
             **dict.fromkeys("dt", "3"), **dict.fromkeys("l", "4"),
             **dict.fromkeys("mn", "5"), **dict.fromkeys("r", "6")}
    head, prev = token[0].upper(), codes.get(token[0], "")
    out = []
    for ch in token[1:]:
        code = codes.get(ch, "")
        if code and code != prev:
            out.append(code)
        if ch not in "hw":
            prev = code
    return (head + "".join(out) + "000")[:4]


def initials_of(name: str) -> str:
    return "".join(t[0] for t in norm_name(name).split() if t)


# --------------------------------------------------------------------------
# field-level similarity
# --------------------------------------------------------------------------

def name_similarity(a: str, b: str) -> float:
    """
    Order-insensitive name comparison that also survives initialisation
    ("Rajesh Kumar Sharma" vs "R. K. Sharma").

    Takes the strongest of three views:
      * token_sort_ratio  - handles reordering / surname-first
      * token_set_ratio   - handles dropped middle names
      * initials view     - handles abbreviated given names
    """
    na, nb = norm_name(a), norm_name(b)
    if not na or not nb:
        return 0.0

    base = max(fuzz.token_sort_ratio(na, nb), fuzz.token_set_ratio(na, nb))

    ta, tb = na.split(), nb.split()
    # Pick the side that is actually initialised ("r k sharma"), not merely the
    # shorter one - both sides often have the same token count.
    a_abbrev = any(len(t) == 1 for t in ta)
    b_abbrev = any(len(t) == 1 for t in tb)
    if a_abbrev != b_abbrev:
        abbrev, full = (ta, tb) if a_abbrev else (tb, ta)
        if abbrev and full:
            surname_score = fuzz.ratio(abbrev[-1], full[-1])
            ia = "".join(t[0] for t in abbrev[:-1])
            ib = "".join(t[0] for t in full[:-1])
            initial_score = 100.0 if ia and ia == ib[:len(ia)] else fuzz.ratio(ia, ib)
            # Surname carries most of the identity signal once given names are
            # reduced to initials.
            base = max(base, 0.7 * surname_score + 0.3 * initial_score)

    return float(base)


def address_similarity(a: str, b: str) -> float:
    na, nb = norm_address(a), norm_address(b)
    if not na or not nb:
        return 0.0
    return float(max(fuzz.token_set_ratio(na, nb), fuzz.token_sort_ratio(na, nb)))


def dob_similarity(a: str, b: str) -> float:
    """Exact = 100, day/month transposition = 90, single-digit slip = 75."""
    if a == b:
        return 100.0
    try:
        ya, ma, da = a.split("-")
        yb, mb, db = b.split("-")
    except ValueError:
        return 0.0
    if ya != yb:
        return 0.0
    if ma == db and da == mb:
        return 90.0
    if (ma == mb) != (da == db):  # exactly one component differs
        return 75.0
    return 20.0


# --------------------------------------------------------------------------
# stage 1: blocking
# --------------------------------------------------------------------------

def build_blocks(df: pd.DataFrame) -> dict[str, list[int]]:
    """
    Map blocking key -> list of dataframe row positions.

    Keys are namespaced by strategy so different strategies never collide.
    Strategies are deliberately redundant: a duplicate only needs to survive
    ONE of them to become a candidate.
    """
    blocks: dict[str, list[int]] = defaultdict(list)

    for pos, row in enumerate(df.itertuples(index=False)):
        name_tokens = norm_name(row.full_name).split()
        surname = name_tokens[-1] if name_tokens else ""
        first = name_tokens[0] if name_tokens else ""
        dob_year = str(row.dob)[:4]

        blocks[f"phone:{norm_phone(row.phone)}"].append(pos)
        blocks[f"pan:{str(row.pan).upper().strip()}"].append(pos)
        blocks[f"email:{norm_email(row.email)}"].append(pos)
        blocks[f"dob:{row.dob}"].append(pos)
        blocks[f"dobcity:{row.dob}|{row.city}"].append(pos)
        blocks[f"sndx_pin:{soundex(surname)}|{row.pincode}"].append(pos)
        blocks[f"sndx_dobyr:{soundex(surname)}|{dob_year}"].append(pos)
        blocks[f"sndx_city:{soundex(surname)}|{soundex(first)}|{row.city}"].append(pos)
        # Order-insensitive full-name key: catches reordered / surname-first dupes.
        blocks[f"namesort:{' '.join(sorted(name_tokens))}"].append(pos)
        blocks[f"initials_city:{''.join(sorted(initials_of(row.full_name)))}|{row.city}"].append(pos)

    return blocks


def candidate_pairs(blocks: dict[str, list[int]],
                    max_block_size: int = MAX_BLOCK_SIZE) -> tuple[set[tuple[int, int]], dict]:
    """Union all within-block pairs, skipping oversized (uninformative) blocks."""
    pairs: set[tuple[int, int]] = set()
    skipped = 0
    for key, members in blocks.items():
        if len(members) < 2:
            continue
        if len(members) > max_block_size:
            skipped += 1
            continue
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))

    stats = {
        "total_blocks": len(blocks),
        "blocks_with_pairs": sum(1 for m in blocks.values() if 2 <= len(m) <= max_block_size),
        "skipped_oversized_blocks": skipped,
        "candidate_pairs": len(pairs),
    }
    return pairs, stats


# --------------------------------------------------------------------------
# stage 2: pair scoring
# --------------------------------------------------------------------------

@dataclass
class PairScore:
    left_id: str
    right_id: str
    confidence: float
    match_type: str
    name_score: float
    address_score: float
    dob_score: float
    phone_exact: bool
    pan_exact: bool
    email_exact: bool

    def as_dict(self) -> dict:
        return self.__dict__


# Weights for the fuzzy blend. Name is the primary identity signal; address and
# DOB corroborate. They sum to 1.0.
W_NAME, W_ADDRESS, W_DOB = 0.50, 0.30, 0.20

# A deterministic identifier hit is treated as near-certain, matching the
# "exact match on phone / PAN / GSTIN" tier in the Razorpay design.
SCORE_PAN_EXACT = 100.0
SCORE_PHONE_EXACT = 97.0
SCORE_EMAIL_EXACT = 95.0


def score_pair(left, right) -> PairScore:
    """Score one candidate pair. `left`/`right` are namedtuple-like rows."""
    pan_exact = str(left.pan).upper().strip() == str(right.pan).upper().strip()
    phone_exact = norm_phone(left.phone) == norm_phone(right.phone)
    email_exact = norm_email(left.email) == norm_email(right.email)

    name_score = name_similarity(left.full_name, right.full_name)
    address_score = address_similarity(left.address, right.address)
    dob_score = dob_similarity(str(left.dob), str(right.dob))

    fuzzy_blend = (W_NAME * name_score
                   + W_ADDRESS * address_score
                   + W_DOB * dob_score)

    if pan_exact:
        confidence, match_type = SCORE_PAN_EXACT, "EXACT_PAN"
    elif phone_exact:
        confidence, match_type = SCORE_PHONE_EXACT, "EXACT_PHONE"
    elif email_exact:
        confidence, match_type = SCORE_EMAIL_EXACT, "EXACT_EMAIL"
    else:
        confidence, match_type = fuzzy_blend, "FUZZY"

    # A strong fuzzy identity match on top of an exact key adds nothing (already
    # ~certain), but a weak one never drags a deterministic hit down.
    return PairScore(
        left_id=left.applicant_id,
        right_id=right.applicant_id,
        confidence=round(float(confidence), 2),
        match_type=match_type,
        name_score=round(name_score, 2),
        address_score=round(address_score, 2),
        dob_score=round(dob_score, 2),
        phone_exact=phone_exact,
        pan_exact=pan_exact,
        email_exact=email_exact,
    )


def run_matching(df: pd.DataFrame, min_confidence: float = 50.0,
                 max_block_size: int = MAX_BLOCK_SIZE) -> tuple[pd.DataFrame, dict]:
    """
    Full pipeline: block -> generate candidates -> score.

    Returns (scored_pairs_df, stats). Pairs below `min_confidence` are dropped
    to keep the output tractable; the tiering thresholds all sit well above it.
    """
    blocks = build_blocks(df)
    pairs, stats = candidate_pairs(blocks, max_block_size)

    rows = list(df.itertuples(index=False))
    scored = []
    for i, j in pairs:
        ps = score_pair(rows[i], rows[j])
        if ps.confidence >= min_confidence:
            scored.append(ps.as_dict())

    out = pd.DataFrame(scored)
    if not out.empty:
        out = out.sort_values("confidence", ascending=False).reset_index(drop=True)

    stats["scored_pairs_retained"] = len(out)
    stats["n_records"] = len(df)
    stats["all_possible_pairs"] = len(df) * (len(df) - 1) // 2
    stats["reduction_ratio"] = round(
        1 - stats["candidate_pairs"] / stats["all_possible_pairs"], 6)
    return out, stats


if __name__ == "__main__":
    import pathlib
    import time

    root = pathlib.Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "data" / "applicants.csv", dtype=str)

    t0 = time.perf_counter()
    pairs_df, stats = run_matching(df)
    elapsed = time.perf_counter() - t0

    (root / "results").mkdir(exist_ok=True)
    pairs_df.to_csv(root / "results" / "scored_pairs.csv", index=False)

    print(f"records                : {stats['n_records']:,}")
    print(f"all possible pairs     : {stats['all_possible_pairs']:,}")
    print(f"candidate pairs        : {stats['candidate_pairs']:,}")
    print(f"search-space reduction : {stats['reduction_ratio'] * 100:.4f}%")
    print(f"skipped oversized blks : {stats['skipped_oversized_blocks']:,}")
    print(f"scored pairs (>=50)    : {stats['scored_pairs_retained']:,}")
    print(f"wall time              : {elapsed:.2f}s")
