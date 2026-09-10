# Duplicate & Fraud Applicant Detection for Loan Underwriting

[![CI](https://github.com/vaishnavigupta-bot/fruad-duplicate-detect-application/actions/workflows/ci.yml/badge.svg)](https://github.com/vaishnavigupta-bot/fruad-duplicate-detect-application/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#license)

An entity-resolution pipeline that finds duplicate and identity-manipulated loan
applicants, scores each match by confidence, and routes only genuinely ambiguous
cases to a human reviewer.

The design follows the architecture Razorpay Capital describes in
[*How does Razorpay Capital detect duplicate or fraud merchants?*](https://engineering.razorpay.com/how-does-razorpay-capital-detect-duplicate-or-fraud-merchants-5ddc67e1535a) —
exact matching on hard identifiers (phone, PAN), fuzzy matching on identity
attributes (name, address), an Elasticsearch layer for real-time screening, and
a good / bad / grey decision split that sends only the grey band to underwriting.

Everything here runs on **synthetic data**. No real applicant information is
used or included.

## Contents

- [Results](#results)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Running it](#running-it)
- [Limitations](#limitations)
- [License](#license)

---

## Results

All figures below are produced by the code in this repo on a fixed seed
(`SEED = 20260908`) and are reproduced end-to-end by CI. Re-run with `make all`.

### Dataset

| | |
|---|---|
| Applicant records | **10,000** |
| Injected duplicate records | 800 (8%) |
| True duplicate pairs | 800 |
| Hard negatives (look-alike but distinct people) | 600 |
| Possible record pairs | 49,995,000 |

Duplicates are injected at three difficulty tiers so recall is a real
measurement rather than an artefact of easy test cases:

| Tier | Records | What changed |
|---|---|---|
| `exact_key` | 360 | Keeps phone **or** PAN; name/address/DOB drift |
| `fuzzy_only` | 320 | **Both** identifiers changed; name/address/DOB perturbed |
| `adversarial` | 120 | Both identifiers changed **and** name restructured (initials, reordering) with the address rewritten |

600 hard negatives are also injected — distinct people sharing a household
address, a common name, or a date of birth — so precision cannot be won trivially.

### 1. Blocking and matching

| Metric | Value |
|---|---|
| Candidate pairs after blocking | 53,353 |
| Search-space reduction | **99.89%** (49,995,000 → 53,353) |
| Blocking recall | **99.88%** (799 / 800 true pairs retained) |
| Batch runtime | **0.72 s** for all 10,000 records |

### 2. Accuracy at the operating threshold

Operating point: **similarity threshold 80**.

| Metric | Value |
|---|---|
| Recall on injected duplicates | **97.9%** (783 / 800) |
| Pair-level precision | 97.5% |
| Pair-level F1 | 0.976 |

Recall by difficulty, showing where the system actually struggles:

| Threshold | `exact_key` | `fuzzy_only` | `adversarial` | **All injected** |
|---|---|---|---|---|
| 80 | 100.0% | 100.0% | 85.8% | **97.9%** |
| 84 | 100.0% | 100.0% | 78.3% | 96.8% |
| 90 | 100.0% | 100.0% | 72.5% | 95.9% |

Best F1 across the sweep is 0.981 at threshold 84, but 80 is the better
operating point for lending: missing a fraudulent duplicate costs more than
sending one extra case to review.

### 3. Three-tier decision system

| Tier | Rule | Applications |
|---|---|---|
| `AUTO_REJECT` | confidence ≥ 92 | 1,492 |
| `FLAG_FOR_REVIEW` | 80 ≤ confidence < 92 | **108** |
| `AUTO_APPROVE` | confidence < 80 | 8,400 |

| Metric | Value |
|---|---|
| Manual review queue, untiered baseline | 1,600 |
| Manual review queue, tiered | **108** |
| **Manual review reduction** | **93.2%** |
| Precision across all flagged matches | **97.8%** |
| Precision of the `AUTO_REJECT` tier | 99.9% |
| Duplicates caught (reject or review) | 97.9% (783 / 800) |
| Duplicates leaked to `AUTO_APPROVE` | 17 |
| False auto-rejects (good customers wrongly denied) | 2 |

**On the baseline.** The comparison is *equal-recall*: a single-threshold system
tuned to the same operating point (80) surfaces the same 1,600 matches but has no
confidence tiering, so every one lands in front of a human. Tiering keeps that
recall and auto-decides the near-certain band. Screening at a looser bar would
make the reduction look better (98.6% against a queue of 7,942) but that baseline
runs at 9.9% precision and no team would operate it, so it is not quoted as the
headline.

### 4. Real-time search layer

Measured against a live single-node **Elasticsearch 8.17.0** (Lucene 9.12.0),
1 shard / 0 replicas, on an Apple Silicon laptop. 1,000 timed queries after 50
warm-up queries, one query per incoming application, `size=10`.

| Metric | Value |
|---|---|
| Documents indexed | 10,000 |
| Bulk indexing throughput | 12,744 docs/sec (0.78 s total) |
| **Mean query latency** | **2.96 ms** |
| p50 / p90 / p95 | 2.84 / 3.74 / 4.03 ms |
| p99 / max | 6.14 / 33.29 ms |
| Server-side `took` (mean) | 2.04 ms |
| Throughput | 338 queries/sec (single client, sequential) |

Retrieval quality — is the true duplicate in the top 5 hits?

| Tier | recall@5 |
|---|---|
| `exact_key` | 100.0% (360/360) |
| `fuzzy_only` | 100.0% (320/320) |
| `adversarial` | 92.5% (111/120) |
| **All injected** | **98.9%** (791/800) |

Latency without retrieval quality is meaningless, which is why recall@5 is
reported alongside it. Interestingly the search layer beats the batch pipeline on
adversarial cases (92.5% vs 85.8%) — BM25 over analysed name and address fields
picks up weak corroborating evidence that the fixed-weight blend discards.

### 5. Scaling past one machine's memory

`matching.py` dedupes a batch against itself, which is the wrong shape once
the existing book no longer fits in memory: it builds every candidate pair
up front. Real underwriting traffic has the opposite shape — one new
application at a time, screened against a book of any size — which is
exactly the single-round-trip ES query above. `src/match_streaming.py`
wires that query to the same field-level scoring (`matching.score_pair`)
and the same tiering thresholds (`decision.assign_tier`), so a screening
decision costs one ES round trip regardless of whether the existing book
holds 10K applicants or 50M. Nothing about the matching logic changes —
only how candidates are generated. Run the demo with `make screen` (needs
a running index — see `make search` below); `tests/test_match_streaming.py`
checks it reaches the same tier as the batch pipeline without needing a
live ES node.

This is an approximation of the batch pipeline, not a guaranteed match —
verified live against 300 real applicants from the shipped dataset, it
reached the same disposition as the batch pipeline for **299/300 (99.7%)**.
The one disagreement happened because ES's top-10 BM25 retrieval didn't
surface the same candidate the batch pipeline's exhaustive blocking found;
exact-identifier hits (PAN/phone/email) always agree, since those don't
depend on retrieval ranking, but fuzzy-only matches near a tier boundary
can occasionally disagree.

---

## How it works

```
 10,000 applicants
        │
        ▼
 ┌──────────────────┐   10 redundant blocking keys: phone, PAN, email, DOB,
 │  Stage 1         │   DOB+city, soundex(surname)+pincode, soundex+DOB-year,
 │  BLOCKING        │   soundex(surname)+soundex(first)+city, sorted-name-tokens,
 └──────────────────┘   initials+city.  A duplicate need survive only ONE.
        │  53,353 candidate pairs (99.89% fewer)
        ▼
 ┌──────────────────┐   exact:  PAN → 100 · phone → 97 · email → 95
 │  Stage 2         │   fuzzy:  0.50·name + 0.30·address + 0.20·DOB
 │  SCORING         │           (RapidFuzz token_sort / token_set, with an
 └──────────────────┘            initials-aware view and a DOB error model)
        │  confidence ∈ [0, 100] per pair
        ▼
 ┌──────────────────┐   ≥ 92  AUTO_REJECT
 │  Stage 3         │   ≥ 80  FLAG_FOR_REVIEW  ─────►  Streamlit console
 │  TIERING         │   < 80  AUTO_APPROVE
 └──────────────────┘
```

### Why these components

**Blocking is redundant on purpose.** Comparing all 50M pairs is wasteful, but a
single blocking key is brittle — an adversarial duplicate that changes its phone
number vanishes from a phone block. Ten overlapping keys mean a record has to
defeat all of them to escape, which is why blocking recall stays at 99.88% while
still discarding 99.89% of the search space.

**Name matching needs more than one view.** `token_sort_ratio` handles
reordering, `token_set_ratio` handles dropped middle names, and neither handles
`Rajesh Kumar Sharma` → `R. K. Sharma`. The engine adds an initials-aware
comparison that weights the surname at 0.7 and the given-name initials at 0.3,
and takes the strongest of the three views.

**DOB is scored with an error model, not equality.** Exact = 100, day/month
transposition = 90, single-digit slip = 75, year mismatch = 0. Real data-entry
errors are not random, and treating a transposed date as a total mismatch throws
away a strong signal.

**Deterministic identifiers short-circuit the blend.** A PAN match is near-proof
of identity; letting a weak address score drag it down would be wrong. Exact hits
bypass the weighted blend entirely.

---

## Repository layout

```
src/
  generate_data.py   synthetic applicant generator with labelled duplicates
                     and hard negatives
  matching.py        normalisation, blocking, field similarity, pair scoring
  evaluate.py        blocking recall, threshold sweep, recall by difficulty
  decision.py        three-tier policy and manual-workload simulation
  search_index.py    Elasticsearch index, dedupe query, latency benchmark
  match_streaming.py per-applicant screening against a live index, any book size
  app.py             Streamlit review console
tests/
  test_matching.py         24 unit tests
  test_match_streaming.py  streaming/batch tier-agreement tests
scripts/
  start_search.sh    launch a local single-node Elasticsearch
results/             generated metrics (JSON + CSV)
```

---

## Running it

Requires Python 3.11+. On a fresh machine:

```bash
make setup    # venv + dependencies
make all      # data -> match -> eval -> decide
make test     # unit tests
```

`make all` is deterministic (fixed seed), so it reproduces every number in the
Results section above on any machine.

For the search benchmark and the dashboard, see the two sections below. The
Elasticsearch distribution is ~450MB and platform-specific, so it is not
committed; `make fetch-search` downloads the correct build for your OS and
architecture.

### Review dashboard

Run it locally:

```bash
make dashboard
```

The console shows the review queue ordered by confidence, an aligned
field-by-field diff with per-field agreement verdicts, the component scores
behind the confidence, the normalised values the matcher actually compared, and
one-click disposition writing to an audit log. (On the hosted demo, the audit
log lives on Streamlit Cloud's ephemeral filesystem — reviewer actions there
don't persist across redeploys; a local `make dashboard` run persists to
`results/review_audit_log.csv` on disk.)

### Search benchmark

Requires a local Elasticsearch:

```bash
make fetch-search          # downloads ~450MB into vendor/ (once)
./scripts/start_search.sh  # start the node
make search                # index + benchmark
./scripts/start_search.sh stop
```

The index uses custom analysers (lowercasing, ASCII folding, address-abbreviation
synonyms) with `.keyword` subfields for the deterministic identifiers. The dedupe
query is a single `bool` combining boosted `term` clauses on PAN/phone/email/DOB
with `fuzziness: AUTO` `match` clauses on name and address, so one round trip
returns a ranked candidate list.

The same index body and query DSL run unchanged on OpenSearch — swap the
`elasticsearch` client for `opensearch-py` and pass `body=` instead of the
keyword arguments.

---

## Limitations

Worth stating plainly, because they bound what the numbers mean.

- **Synthetic data.** Perturbations are drawn from a hand-built model of
  data-entry error (typos, phonetic variants, abbreviations, initialisation).
  Real applicant data is messier and its error distribution is not known in
  advance, so real-world recall would differ.
- **Thresholds are tuned on the same dataset they are evaluated on.** There is no
  held-out split. The thresholds are chosen from the sweep for interpretability,
  not fitted, but the numbers are still in-sample.
- **The scoring weights are hand-set, not learned.** 0.50/0.30/0.20 across
  name/address/DOB is a reasonable prior, not an optimum. A labelled production
  dataset would justify a learned model.
- **Adversarial recall is 85.8%, not 99%.** Duplicates that change both
  identifiers *and* restructure the name and address are genuinely hard. This is
  the honest ceiling of string-similarity matching without device fingerprints,
  bureau pulls, or graph signals.
- **Latency is measured single-client and sequential** against a single-shard,
  single-node index on a laptop — it is a clean relative measure, not a
  production capacity figure.

## License

MIT
