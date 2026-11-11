"""
NSAP Dataset Generator
======================
Generates a realistic synthetic dataset that mirrors the structure of the
AI Kosh NSAP district-wise pension dataset.

NSAP Sub-schemes (target classes):
  0 – IGNOAPS : Indira Gandhi National Old Age Pension Scheme
  1 – IGNWPS  : Indira Gandhi National Widow Pension Scheme
  2 – IGNDPS  : Indira Gandhi National Disability Pension Scheme
  3 – NFBS    : National Family Benefit Scheme

Eligibility rules encoded in generation logic:
  IGNOAPS : age >= 60, BPL household, not disabled, not widow (primary)
  IGNWPS  : female, widow=True, age 18-79, BPL household, not disabled
  IGNDPS  : disability_pct >= 80, age 18-79, BPL household
  NFBS    : primary earner death, family age 18-64, BPL household
"""

import numpy as np
import pandas as pd

STATES = [
    "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Gujarat",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Odisha", "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh",
    "West Bengal", "Himachal Pradesh", "Uttarakhand", "Punjab", "Haryana",
]

DISTRICTS_PER_STATE = {s: [f"{s[:4].strip()}_D{i}" for i in range(1, 6)] for s in STATES}

SCHEME_LABELS = {0: "IGNOAPS", 1: "IGNWPS", 2: "IGNDPS", 3: "NFBS"}
SCHEME_CODES  = {v: k for k, v in SCHEME_LABELS.items()}


def _assign_scheme(row: dict) -> int:
    """Rule-based assignment of NSAP scheme – mirrors official eligibility."""
    age       = row["age"]
    gender    = row["gender"]           # 0=Male, 1=Female
    widow     = row["is_widow"]
    disabled  = row["is_disabled"]
    dis_pct   = row["disability_percentage"]
    bpl       = row["bpl_card"]
    earner_d  = row["primary_earner_deceased"]

    if not bpl:
        return -1  # ineligible – excluded from dataset

    # Priority order: IGNDPS > IGNWPS > IGNOAPS > NFBS
    if disabled and dis_pct >= 80 and 18 <= age <= 79:
        return 2   # IGNDPS
    if gender == 1 and widow and 18 <= age <= 79:
        return 1   # IGNWPS
    if age >= 60:
        return 0   # IGNOAPS
    if earner_d and 18 <= age <= 64:
        return 3   # NFBS
    return -1  # ineligible


def generate_dataset(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic NSAP applicant dataset."""
    rng = np.random.default_rng(random_state)
    records = []

    target_per_class = n_samples // 4
    # Generate with slight imbalance (realistic)
    class_counts = {0: int(target_per_class * 1.4), 1: int(target_per_class * 1.0),
                    2: int(target_per_class * 0.7), 3: int(target_per_class * 0.9)}

    for scheme_id, count in class_counts.items():
        for _ in range(count):
            state   = rng.choice(STATES)
            district = rng.choice(DISTRICTS_PER_STATE[state])

            if scheme_id == 0:   # IGNOAPS – old age
                age    = int(rng.integers(60, 90))
                gender = int(rng.choice([0, 1], p=[0.55, 0.45]))
                widow  = bool(gender == 1 and rng.random() < 0.25)
                dis    = bool(rng.random() < 0.05)
                dis_p  = int(rng.integers(10, 40)) if dis else 0
                earner = bool(rng.random() < 0.05)

            elif scheme_id == 1:  # IGNWPS – widow
                age    = int(rng.integers(18, 79))
                gender = 1
                widow  = True
                dis    = bool(rng.random() < 0.03)
                dis_p  = int(rng.integers(10, 40)) if dis else 0
                earner = bool(rng.random() < 0.1)

            elif scheme_id == 2:  # IGNDPS – disability
                age    = int(rng.integers(18, 79))
                gender = int(rng.choice([0, 1], p=[0.55, 0.45]))
                widow  = bool(gender == 1 and rng.random() < 0.05)
                dis    = True
                dis_p  = int(rng.integers(80, 100))
                earner = bool(rng.random() < 0.05)

            else:                  # NFBS – family benefit
                age    = int(rng.integers(18, 64))
                gender = int(rng.choice([0, 1], p=[0.45, 0.55]))
                widow  = bool(gender == 1 and rng.random() < 0.3)
                dis    = bool(rng.random() < 0.03)
                dis_p  = int(rng.integers(10, 40)) if dis else 0
                earner = True

            row = {
                "state"                    : state,
                "district"                 : district,
                "age"                      : age,
                "gender"                   : gender,          # 0=M, 1=F
                "is_widow"                 : int(widow),
                "is_disabled"              : int(dis),
                "disability_percentage"    : dis_p,
                "bpl_card"                 : 1,               # all BPL
                "primary_earner_deceased"  : int(earner),
                "annual_income"            : int(rng.integers(10000, 90000)),
                "land_holding_acres"       : round(float(rng.uniform(0, 2.5)), 2),
                "household_size"           : int(rng.integers(1, 9)),
                "education_level"          : int(rng.choice([0, 1, 2, 3],
                                                p=[0.45, 0.30, 0.15, 0.10])),
                "ration_card_type"         : int(rng.choice([0, 1, 2],
                                                p=[0.60, 0.25, 0.15])),
                "scheme"                   : scheme_id,
            }
            records.append(row)

    df = pd.DataFrame(records).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Apply noise / edge cases (~3% label noise to make it realistic)
    noise_idx = rng.choice(len(df), size=int(len(df) * 0.03), replace=False)
    df.loc[noise_idx, "scheme"] = rng.integers(0, 4, size=len(noise_idx))

    df["scheme_name"] = df["scheme"].map(SCHEME_LABELS)
    return df


def load_or_generate(path: str = "data/nsap_dataset.csv",
                     n_samples: int = 10000) -> pd.DataFrame:
    """Load existing CSV or generate and save a new one."""
    import os
    if os.path.exists(path):
        print(f"[DataGenerator] Loading existing dataset from {path}")
        return pd.read_csv(path)
    print(f"[DataGenerator] Generating {n_samples} synthetic NSAP records …")
    df = generate_dataset(n_samples)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[DataGenerator] Saved to {path}  shape={df.shape}")
    return df


if __name__ == "__main__":
    df = load_or_generate()
    print(df.head())
    print("\nScheme distribution:")
    print(df["scheme_name"].value_counts())
