"""
Synthetic loan-applicant dataset generator.

Produces a 10,000-record applicant table with a known ground-truth duplicate
labelling, modelled on the entity-resolution problem described in Razorpay
Capital's merchant-dedupe post: exact identifiers (phone, PAN) plus fuzzy
identity attributes (name, address).

Design goals
------------
1. Duplicates are injected at three difficulty tiers so that measured recall is
   meaningful rather than trivially 100%.
2. HARD NEGATIVES are injected deliberately: distinct people who share a
   surname, a household address, or a date of birth. Without these, precision
   is meaningless because every non-duplicate pair would be obviously distinct.

Every record carries `dup_group_id`: records sharing a group id are the same
real person. Singletons get a unique group id of their own.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, asdict

import pandas as pd

SEED = 20260908
N_RECORDS = 10_000

# Difficulty mix across injected duplicate records.
#   exact_key : shares phone OR PAN with its original -> deterministic match
#   fuzzy_only: both identifiers changed, identity attributes perturbed
#   adversarial: both identifiers changed AND name/address heavily rewritten
DUP_MIX = {"exact_key": 0.45, "fuzzy_only": 0.40, "adversarial": 0.15}
DUP_RECORD_FRACTION = 0.08  # ~800 of 10,000 records are injected duplicates

FIRST_NAMES = [
    "Aarav", "Aditya", "Akshay", "Amit", "Ananya", "Anil", "Anjali", "Ankit",
    "Arjun", "Arun", "Ashwin", "Bhavna", "Chetan", "Deepak", "Deepika",
    "Dhruv", "Divya", "Farhan", "Gaurav", "Harish", "Ishaan", "Jyoti",
    "Kavita", "Kiran", "Lakshmi", "Manish", "Meera", "Mohit", "Naveen",
    "Neha", "Nikhil", "Nisha", "Pooja", "Prakash", "Pranav", "Priya",
    "Rahul", "Rajesh", "Rakesh", "Ramesh", "Ravi", "Rekha", "Rohit",
    "Sandeep", "Sanjay", "Saurabh", "Shalini", "Shreya", "Siddharth",
    "Sneha", "Sunil", "Suresh", "Swati", "Tanvi", "Uday", "Varun", "Vikas",
    "Vikram", "Vinod", "Vishal", "Yash", "Zoya",
]

MIDDLE_NAMES = ["Kumar", "Chandra", "Prasad", "Devi", "Nath", "Lal", "Rani", ""]

LAST_NAMES = [
    "Agarwal", "Bansal", "Bhat", "Chauhan", "Chopra", "Desai", "Dixit",
    "Gupta", "Iyer", "Jain", "Joshi", "Kapoor", "Khanna", "Kulkarni",
    "Malhotra", "Mehta", "Menon", "Mishra", "Nair", "Pandey", "Patel",
    "Pillai", "Rao", "Reddy", "Saxena", "Sharma", "Shetty", "Shukla",
    "Singh", "Sinha", "Tiwari", "Trivedi", "Verma", "Yadav",
]

# Real-world spelling variance for the same spoken name. These are the cases a
# pure exact-match system misses entirely.
PHONETIC_VARIANTS = {
    "Sanjay": "Sanjai", "Rajesh": "Rajish", "Ramesh": "Ramesh",
    "Neha": "Neeha", "Priya": "Priyaa", "Rohit": "Rohith",
    "Sunil": "Suneel", "Vikram": "Bikram", "Deepak": "Dipak",
    "Ashwin": "Aswin", "Siddharth": "Sidharth", "Saurabh": "Sourabh",
    "Kavita": "Kavitha", "Swati": "Swathi", "Sharma": "Sarma",
    "Verma": "Varma", "Mishra": "Misra", "Chauhan": "Chouhan",
    "Bhat": "Bhatt", "Shetty": "Setty", "Reddy": "Reddi",
}

CITIES = [
    ("Mumbai", "Maharashtra", "400"), ("Pune", "Maharashtra", "411"),
    ("Bengaluru", "Karnataka", "560"), ("Mysuru", "Karnataka", "570"),
    ("Chennai", "Tamil Nadu", "600"), ("Coimbatore", "Tamil Nadu", "641"),
    ("Hyderabad", "Telangana", "500"), ("Delhi", "Delhi", "110"),
    ("Gurugram", "Haryana", "122"), ("Noida", "Uttar Pradesh", "201"),
    ("Jaipur", "Rajasthan", "302"), ("Ahmedabad", "Gujarat", "380"),
    ("Surat", "Gujarat", "395"), ("Kolkata", "West Bengal", "700"),
    ("Lucknow", "Uttar Pradesh", "226"), ("Indore", "Madhya Pradesh", "452"),
    ("Kochi", "Kerala", "682"), ("Bhopal", "Madhya Pradesh", "462"),
]

STREET_WORDS = [
    "MG Road", "Nehru Street", "Gandhi Marg", "Link Road", "Ring Road",
    "Church Street", "Station Road", "Park Avenue", "Hill View Road",
    "Lake Side Road", "Temple Street", "Market Road", "Bazaar Street",
]

LOCALITIES = [
    "Andheri East", "Koramangala", "Indiranagar", "Salt Lake", "Adyar",
    "Banjara Hills", "Vasant Kunj", "Sector 45", "Malviya Nagar",
    "Jubilee Hills", "Hinjewadi", "Whitefield", "Powai", "Kothrud",
    "Rajouri Garden", "Anna Nagar", "HSR Layout", "Aundh",
]

EMAIL_DOMAINS = ["gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com", "hotmail.com"]

LOAN_PURPOSES = [
    "Working Capital", "Business Expansion", "Equipment Purchase",
    "Inventory Finance", "Personal Loan", "Debt Consolidation",
]

# Abbreviation pairs applied to addresses to simulate free-text entry variance.
ADDRESS_ABBREV = [
    ("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave"), ("Marg", "Marg."),
    ("Apartment", "Apt"), ("Building", "Bldg"), ("Floor", "Flr"),
    ("Near", "Nr"), ("Opposite", "Opp"), ("Sector", "Sec"),
    ("East", "E"), ("West", "W"), ("North", "N"), ("South", "S"),
]

KEYBOARD_NEIGHBOURS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg",
    "g": "fh", "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "k",
    "m": "n", "n": "bm", "o": "ip", "p": "o", "q": "wa", "r": "et",
    "s": "ad", "t": "ry", "u": "yi", "v": "cb", "w": "qe", "x": "zc",
    "y": "tu", "z": "x",
}


@dataclass
class Applicant:
    applicant_id: str
    dup_group_id: int
    record_kind: str          # 'original' | 'exact_key' | 'fuzzy_only' | 'adversarial'
    full_name: str
    dob: str
    phone: str
    pan: str
    email: str
    address: str
    city: str
    state: str
    pincode: str
    loan_amount: int
    loan_purpose: str
    applied_on: str


# --------------------------------------------------------------------------
# primitive field generators
# --------------------------------------------------------------------------

def _rand_phone(rng: random.Random) -> str:
    return f"{rng.choice('6789')}{''.join(rng.choice('0123456789') for _ in range(9))}"


def _rand_pan(rng: random.Random) -> str:
    """PAN format: 5 letters, 4 digits, 1 letter (e.g. ABCDE1234F)."""
    letters = string.ascii_uppercase
    return (
        "".join(rng.choice(letters) for _ in range(5))
        + "".join(rng.choice("0123456789") for _ in range(4))
        + rng.choice(letters)
    )


def _rand_dob(rng: random.Random) -> str:
    year = rng.randint(1965, 2002)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def _rand_address(rng: random.Random) -> str:
    house = rng.randint(1, 400)
    unit = rng.choice(["", f"Flat {rng.randint(101, 1204)}, ", f"{rng.randint(1, 18)}th Floor, "])
    return f"{unit}{house} {rng.choice(STREET_WORDS)}, {rng.choice(LOCALITIES)}"


def _email_for(name: str, rng: random.Random) -> str:
    handle = name.lower().replace(".", "").replace(" ", rng.choice(["", ".", "_"]))
    return f"{handle}{rng.randint(1, 999)}@{rng.choice(EMAIL_DOMAINS)}"


def _rand_date_2025(rng: random.Random) -> str:
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"2025-{month:02d}-{day:02d}"


# --------------------------------------------------------------------------
# perturbation operators (what makes a duplicate "fuzzy")
# --------------------------------------------------------------------------

def _typo(token: str, rng: random.Random) -> str:
    """Apply one character-level typo: transpose, drop, or keyboard-substitute."""
    if len(token) < 4:
        return token
    op = rng.choice(["transpose", "drop", "substitute", "double"])
    i = rng.randint(1, len(token) - 2)
    if op == "transpose":
        return token[:i] + token[i + 1] + token[i] + token[i + 2:]
    if op == "drop":
        return token[:i] + token[i + 1:]
    if op == "double":
        return token[:i] + token[i] + token[i:]
    ch = token[i].lower()
    if ch in KEYBOARD_NEIGHBOURS:
        return token[:i] + rng.choice(KEYBOARD_NEIGHBOURS[ch]) + token[i + 1:]
    return token


def perturb_name(name: str, rng: random.Random, level: str) -> str:
    """
    level='light'  -> one small change (typo or phonetic variant)
    level='heavy'  -> structural change (initials, reorder, dropped middle name)
    """
    parts = name.split()
    if level == "light":
        op = rng.choice(["typo", "phonetic", "phonetic", "drop_middle"])
        if op == "phonetic":
            for idx, p in enumerate(parts):
                if p in PHONETIC_VARIANTS:
                    parts[idx] = PHONETIC_VARIANTS[p]
                    return " ".join(parts)
            op = "typo"  # no variant available, fall through
        if op == "drop_middle" and len(parts) == 3:
            return f"{parts[0]} {parts[2]}"
        idx = rng.randrange(len(parts))
        parts[idx] = _typo(parts[idx], rng)
        return " ".join(parts)

    # heavy: the evasive-rewrite cases
    op = rng.choice(["initials", "reorder", "initial_plus_typo", "surname_first"])
    if op == "initials" and len(parts) >= 2:
        return " ".join(f"{p[0]}." for p in parts[:-1]) + f" {parts[-1]}"
    if op == "reorder" and len(parts) >= 2:
        return " ".join(reversed(parts))
    if op == "surname_first" and len(parts) >= 2:
        return f"{parts[-1]} {' '.join(parts[:-1])}"
    # initial_plus_typo
    head = f"{parts[0][0]}."
    tail = _typo(parts[-1], rng)
    return f"{head} {tail}"


def perturb_address(address: str, rng: random.Random, level: str) -> str:
    out = address
    if level == "light":
        for full, abbr in rng.sample(ADDRESS_ABBREV, k=3):
            out = out.replace(full, abbr)
        if rng.random() < 0.4:
            out = out.replace(",", "")
        return out.strip()

    # heavy: drop the unit/house prefix and abbreviate aggressively
    segments = [s.strip() for s in out.split(",")]
    if len(segments) > 1 and rng.random() < 0.7:
        segments = segments[1:]
    out = ", ".join(segments)
    for full, abbr in ADDRESS_ABBREV:
        out = out.replace(full, abbr)
    if rng.random() < 0.5:
        out = out.upper()
    return out.strip()


def perturb_dob(dob: str, rng: random.Random) -> str:
    """Day/month transposition or a single-digit slip — common data-entry error."""
    y, m, d = dob.split("-")
    op = rng.choice(["swap_dm", "digit", "same", "same"])
    if op == "same":
        return dob
    if op == "swap_dm" and int(d) <= 12:
        return f"{y}-{d}-{m}"
    return f"{y}-{m}-{int(d) % 28 + 1:02d}"


def perturb_phone(phone: str, rng: random.Random) -> str:
    """Change 1-2 digits: still a different number, so exact match must fail."""
    chars = list(phone)
    for i in rng.sample(range(1, 10), k=rng.choice([1, 2])):
        chars[i] = rng.choice([c for c in "0123456789" if c != chars[i]])
    return "".join(chars)


# --------------------------------------------------------------------------
# record construction
# --------------------------------------------------------------------------

def make_original(rng: random.Random, group_id: int, seq: int,
                  force: dict | None = None) -> Applicant:
    first = rng.choice(FIRST_NAMES)
    middle = rng.choice(MIDDLE_NAMES)
    last = rng.choice(LAST_NAMES)
    name = " ".join(p for p in (first, middle, last) if p)
    city, state, pin_prefix = rng.choice(CITIES)
    pincode = f"{pin_prefix}{rng.randint(0, 99):03d}"[:6].ljust(6, "0")

    rec = Applicant(
        applicant_id=f"APP{seq:06d}",
        dup_group_id=group_id,
        record_kind="original",
        full_name=name,
        dob=_rand_dob(rng),
        phone=_rand_phone(rng),
        pan=_rand_pan(rng),
        email=_email_for(name, rng),
        address=_rand_address(rng),
        city=city,
        state=state,
        pincode=pincode,
        loan_amount=rng.randrange(50_000, 5_000_000, 25_000),
        loan_purpose=rng.choice(LOAN_PURPOSES),
        applied_on=_rand_date_2025(rng),
    )
    if force:
        for k, v in force.items():
            setattr(rec, k, v)
    return rec


def make_duplicate(src: Applicant, rng: random.Random, kind: str, seq: int) -> Applicant:
    """Build a duplicate of `src` at the requested difficulty tier."""
    dup = Applicant(**asdict(src))
    dup.applicant_id = f"APP{seq:06d}"
    dup.record_kind = kind
    dup.loan_amount = rng.randrange(50_000, 5_000_000, 25_000)
    dup.loan_purpose = rng.choice(LOAN_PURPOSES)
    dup.applied_on = _rand_date_2025(rng)

    if kind == "exact_key":
        # Retains at least one hard identifier; other fields drift.
        keep = rng.choice(["phone", "pan", "both"])
        if keep == "phone":
            dup.pan = _rand_pan(rng)
        elif keep == "pan":
            dup.phone = perturb_phone(src.phone, rng)
        dup.full_name = perturb_name(src.full_name, rng, "light")
        dup.address = perturb_address(src.address, rng, "light")
        dup.dob = perturb_dob(src.dob, rng)

    elif kind == "fuzzy_only":
        # Both hard identifiers differ -> only fuzzy logic can catch this.
        dup.phone = perturb_phone(src.phone, rng)
        dup.pan = _rand_pan(rng)
        dup.full_name = perturb_name(src.full_name, rng, "light")
        dup.address = perturb_address(src.address, rng, "light")
        dup.dob = perturb_dob(src.dob, rng)

    else:  # adversarial
        # Deliberate evasion: new identifiers, restructured name and address.
        dup.phone = _rand_phone(rng)
        dup.pan = _rand_pan(rng)
        dup.full_name = perturb_name(src.full_name, rng, "heavy")
        dup.address = perturb_address(src.address, rng, "heavy")
        dup.dob = src.dob if rng.random() < 0.75 else perturb_dob(src.dob, rng)
        if rng.random() < 0.3:  # sometimes relocates within the same city
            dup.address = _rand_address(rng)

    dup.email = _email_for(dup.full_name, rng)
    return dup


def make_hard_negative(src: Applicant, rng: random.Random, group_id: int,
                       seq: int, flavour: str) -> Applicant:
    """
    A genuinely DIFFERENT person who looks similar to `src`. These exist to make
    precision a real measurement.

    flavour='household' -> different person, same address (spouse/sibling/PG)
    flavour='namesake'  -> same common name, different city and identifiers
    flavour='dob_twin'  -> same DOB and city, unrelated identity
    """
    rec = make_original(rng, group_id, seq)
    rec.record_kind = f"hard_negative_{flavour}"

    if flavour == "household":
        rec.address = src.address
        rec.city, rec.state, rec.pincode = src.city, src.state, src.pincode
        parts = src.full_name.split()
        rec.full_name = f"{rng.choice(FIRST_NAMES)} {parts[-1]}"  # shares surname
    elif flavour == "namesake":
        rec.full_name = src.full_name  # identical name, different person
    else:  # dob_twin
        rec.dob = src.dob
        rec.city, rec.state, rec.pincode = src.city, src.state, src.pincode

    rec.email = _email_for(rec.full_name, rng)
    return rec


def generate(n_records: int = N_RECORDS, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)

    n_dupes = int(n_records * DUP_RECORD_FRACTION)
    n_hard_neg = int(n_records * 0.06)
    n_originals = n_records - n_dupes - n_hard_neg

    records: list[Applicant] = []
    seq = 0
    group = 0

    # 1. base population
    for _ in range(n_originals):
        records.append(make_original(rng, group, seq))
        seq += 1
        group += 1

    # 2. injected duplicates, distributed across difficulty tiers
    kinds = (
        ["exact_key"] * round(n_dupes * DUP_MIX["exact_key"])
        + ["fuzzy_only"] * round(n_dupes * DUP_MIX["fuzzy_only"])
        + ["adversarial"] * round(n_dupes * DUP_MIX["adversarial"])
    )
    kinds = kinds[:n_dupes] + ["fuzzy_only"] * (n_dupes - len(kinds))
    rng.shuffle(kinds)

    # Duplicate sources are sampled without replacement so most groups are pairs;
    # a minority get a second duplicate, producing triples.
    sources = rng.sample(records, k=min(n_dupes, len(records)))
    for kind, src in zip(kinds, sources):
        records.append(make_duplicate(src, rng, kind, seq))
        seq += 1

    # 3. hard negatives anchored on random existing records
    flavours = (["household"] * (n_hard_neg // 2)
                + ["namesake"] * (n_hard_neg // 4)
                + ["dob_twin"] * (n_hard_neg - n_hard_neg // 2 - n_hard_neg // 4))
    for flavour in flavours:
        anchor = rng.choice(records)
        records.append(make_hard_negative(anchor, rng, group, seq, flavour))
        seq += 1
        group += 1

    rng.shuffle(records)
    df = pd.DataFrame([asdict(r) for r in records])
    # Re-issue ids in shuffled order so id ordering leaks no ground truth.
    df["applicant_id"] = [f"APP{i:06d}" for i in range(len(df))]
    return df


if __name__ == "__main__":
    import pathlib

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate()
    path = out_dir / "applicants.csv"
    df.to_csv(path, index=False)

    dup_records = df[df.record_kind.isin(DUP_MIX.keys())]
    group_sizes = df.dup_group_id.value_counts()

    print(f"wrote {path}  ({len(df):,} records)")
    print("\nrecord_kind breakdown")
    print(df.record_kind.value_counts().to_string())
    print(f"\ninjected duplicate records : {len(dup_records):,}")
    print(f"duplicate groups (size > 1) : {(group_sizes > 1).sum():,}")
    print(f"true duplicate PAIRS        : "
          f"{int((group_sizes * (group_sizes - 1) // 2).sum()):,}")
    print(f"total candidate pairs (n^2) : {len(df) * (len(df) - 1) // 2:,}")
