"""
Underwriter review dashboard.

Purpose: make a flagged application decidable in seconds instead of minutes.
Without this, an analyst reconciling a possible duplicate has to pull both
records from the source system, put them side by side, and eyeball every field.

The dashboard collapses that into one screen:
  * the review queue, ordered by confidence
  * a field-by-field diff of the two records, with agreement colour-coded
  * the component scores that produced the confidence, so the machine's
    reasoning is inspectable rather than a black box
  * one-click disposition, appended to an audit log

Run:  streamlit run src/app.py
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from matching import (address_similarity, dob_similarity, name_similarity,
                      norm_address, norm_name, norm_phone)

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUDIT_LOG = ROOT / "results" / "review_audit_log.csv"

st.set_page_config(page_title="Applicant Dedupe Review",
                   page_icon="🔍", layout="wide")


@st.cache_data
def load():
    applicants = pd.read_csv(ROOT / "data" / "applicants.csv", dtype=str)
    decisions = pd.read_csv(ROOT / "results" / "application_decisions.csv", dtype=str)
    decisions["confidence"] = decisions["confidence"].astype(float)
    for col in ("name_score", "address_score", "dob_score"):
        decisions[col] = decisions[col].astype(float)
    return applicants.set_index("applicant_id"), decisions


def field_rows(left: pd.Series, right: pd.Series) -> list[dict]:
    """Build the side-by-side comparison, scoring agreement per field."""
    def verdict(score: float) -> str:
        if score >= 95:
            return "MATCH"
        if score >= 75:
            return "NEAR"
        return "DIFFER"

    specs = [
        ("Full name", "full_name",
         lambda a, b: name_similarity(a, b)),
        ("Date of birth", "dob",
         lambda a, b: dob_similarity(a, b)),
        ("Phone", "phone",
         lambda a, b: 100.0 if norm_phone(a) == norm_phone(b) else 0.0),
        ("PAN", "pan",
         lambda a, b: 100.0 if str(a).upper() == str(b).upper() else 0.0),
        ("Email", "email",
         lambda a, b: 100.0 if str(a).lower() == str(b).lower() else 0.0),
        ("Address", "address",
         lambda a, b: address_similarity(a, b)),
        ("City", "city", lambda a, b: 100.0 if a == b else 0.0),
        ("Pincode", "pincode", lambda a, b: 100.0 if a == b else 0.0),
    ]

    rows = []
    for label, col, scorer in specs:
        a, b = str(left[col]), str(right[col])
        score = float(scorer(a, b))
        rows.append({
            "Field": label,
            "Application A": a,
            "Application B": b,
            "Similarity": round(score, 1),
            "Verdict": verdict(score),
        })
    return rows


def style_verdict(row):
    # Light backgrounds with explicit dark text: the dataframe renders on
    # Streamlit's light theme, so dark fills would leave text unreadable.
    colours = {
        "MATCH":  ("#d1e7dd", "#0a3622"),
        "NEAR":   ("#fff3cd", "#664d03"),
        "DIFFER": ("#f8d7da", "#58151c"),
    }
    bg, fg = colours.get(row["Verdict"], ("", "inherit"))
    return [f"background-color: {bg}; color: {fg}"] * len(row)


def record_decision(app_id: str, matched_id: str, confidence: float,
                    disposition: str, note: str) -> None:
    entry = {
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "applicant_id": app_id,
        "matched_id": matched_id,
        "confidence": confidence,
        "disposition": disposition,
        "note": note,
    }
    AUDIT_LOG.parent.mkdir(exist_ok=True)
    header = not AUDIT_LOG.exists()
    pd.DataFrame([entry]).to_csv(AUDIT_LOG, mode="a", header=header, index=False)


# --------------------------------------------------------------------------

applicants, decisions = load()

st.title("🔍 Loan Applicant Deduplication — Review Console")

# ---- portfolio-level summary ---------------------------------------------
tier_counts = decisions.tier.value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Applications screened", f"{len(decisions):,}")
c2.metric("Auto-rejected", f"{tier_counts.get('AUTO_REJECT', 0):,}",
          help="Confidence ≥ 92 — treated as a confirmed duplicate.")
c3.metric("Flagged for review", f"{tier_counts.get('FLAG_FOR_REVIEW', 0):,}",
          help="Grey zone (80 ≤ confidence < 92) — needs a human.")
c4.metric("Auto-approved", f"{tier_counts.get('AUTO_APPROVE', 0):,}",
          help="No credible duplicate found.")

bench_path = ROOT / "results" / "search_benchmark.json"
if bench_path.exists():
    bench = json.loads(bench_path.read_text())
    lat = bench["benchmark"]["client_side_latency_ms"]
    st.caption(
        f"{bench['engine']} · {bench['benchmark']['indexed_documents']:,} documents "
        f"indexed · mean query latency {lat['mean']} ms · p95 {lat['p95']} ms · "
        f"recall@{bench['retrieval']['top_k']} "
        f"{bench['retrieval']['by_kind']['ALL']['recall_at_k'] * 100:.1f}%"
    )

st.divider()

# ---- queue ----------------------------------------------------------------
with st.sidebar:
    st.header("Review queue")
    tier = st.selectbox("Tier", ["FLAG_FOR_REVIEW", "AUTO_REJECT", "AUTO_APPROVE"])
    min_conf, max_conf = st.slider("Confidence range", 0.0, 100.0, (0.0, 100.0), 0.5)
    match_types = sorted(decisions.match_type.unique())
    chosen_types = st.multiselect("Match type", match_types, default=match_types)

queue = decisions[
    (decisions.tier == tier)
    & decisions.confidence.between(min_conf, max_conf)
    & decisions.match_type.isin(chosen_types)
].sort_values("confidence", ascending=False)

with st.sidebar:
    st.write(f"**{len(queue):,}** cases in queue")

if queue.empty:
    st.info("No cases match the current filters.")
    st.stop()

labels = [
    f"{r.applicant_id} ↔ {r.matched_id}  ·  {r.confidence:.1f}  ·  {r.match_type}"
    for r in queue.itertuples()
]
choice = st.sidebar.radio("Select a case", labels, index=0, label_visibility="collapsed")
case = queue.iloc[labels.index(choice)]

# ---- case detail ----------------------------------------------------------
left_id, right_id = case.applicant_id, case.matched_id
if right_id not in applicants.index:
    st.warning("Matched record not found.")
    st.stop()

left, right = applicants.loc[left_id], applicants.loc[right_id]

head1, head2 = st.columns([3, 2])
with head1:
    st.subheader(f"Case  {left_id}  ↔  {right_id}")
    st.write(f"**Match type:** `{case.match_type}`  ·  "
             f"**Tier:** `{case.tier}`")
with head2:
    st.metric("Match confidence", f"{case.confidence:.1f} / 100")

s1, s2, s3 = st.columns(3)
s1.progress(min(case.name_score / 100, 1.0), text=f"Name {case.name_score:.0f}")
s2.progress(min(case.address_score / 100, 1.0), text=f"Address {case.address_score:.0f}")
s3.progress(min(case.dob_score / 100, 1.0), text=f"DOB {case.dob_score:.0f}")

st.markdown("#### Field-by-field comparison")
cmp_df = pd.DataFrame(field_rows(left, right))
st.dataframe(
    cmp_df.style.apply(style_verdict, axis=1),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Similarity": st.column_config.NumberColumn("Similarity", format="%.1f"),
    },
)

with st.expander("Loan context"):
    ctx = pd.DataFrame({
        "Field": ["Loan amount", "Purpose", "Applied on"],
        "Application A": [f"₹{int(left.loan_amount):,}", left.loan_purpose, left.applied_on],
        "Application B": [f"₹{int(right.loan_amount):,}", right.loan_purpose, right.applied_on],
    })
    st.dataframe(ctx, use_container_width=True, hide_index=True)

with st.expander("Normalised values (what the matcher actually compared)"):
    st.json({
        "name":    {"A": norm_name(left.full_name),   "B": norm_name(right.full_name)},
        "address": {"A": norm_address(left.address),  "B": norm_address(right.address)},
        "phone":   {"A": norm_phone(left.phone),      "B": norm_phone(right.phone)},
    })

# ---- disposition ----------------------------------------------------------
st.markdown("#### Disposition")
note = st.text_input("Reviewer note (optional)", key=f"note_{left_id}")
d1, d2, d3 = st.columns(3)
if d1.button("🚫 Confirm duplicate — reject", use_container_width=True):
    record_decision(left_id, right_id, case.confidence, "REJECTED_DUPLICATE", note)
    st.success(f"{left_id} rejected as duplicate of {right_id}.")
if d2.button("✅ Not a duplicate — approve", use_container_width=True):
    record_decision(left_id, right_id, case.confidence, "APPROVED_DISTINCT", note)
    st.success(f"{left_id} approved as a distinct applicant.")
if d3.button("⏫ Escalate", use_container_width=True):
    record_decision(left_id, right_id, case.confidence, "ESCALATED", note)
    st.info(f"{left_id} escalated.")

if AUDIT_LOG.exists():
    with st.expander("Audit log"):
        st.dataframe(pd.read_csv(AUDIT_LOG).tail(50),
                     use_container_width=True, hide_index=True)
