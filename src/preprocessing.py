"""
Preprocessing Pipeline for NSAP Scheme Classification
======================================================
Handles feature engineering, encoding, scaling, and train/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# ── Column definitions ──────────────────────────────────────────────────────
CATEGORICAL_COLS = ["state", "district", "ration_card_type", "education_level"]
NUMERICAL_COLS   = [
    "age", "gender", "is_widow", "is_disabled", "disability_percentage",
    "bpl_card", "primary_earner_deceased", "annual_income",
    "land_holding_acres", "household_size",
]
TARGET_COL       = "scheme"
DROP_COLS        = ["scheme_name"]   # human-readable label – not a feature


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[Preprocessing] Loaded {len(df)} records, {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: remove invalid schemes, handle missing values."""
    df = df.copy()

    # Drop rows with invalid scheme labels (noise artefact)
    df = df[df[TARGET_COL].isin([0, 1, 2, 3])].reset_index(drop=True)

    # Fill rare NaN values
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Clamp outliers
    df["age"]                    = df["age"].clip(0, 120)
    df["disability_percentage"]  = df["disability_percentage"].clip(0, 100)
    df["annual_income"]          = df["annual_income"].clip(0, 500000)
    df["land_holding_acres"]     = df["land_holding_acres"].clip(0, 50)

    print(f"[Preprocessing] After cleaning: {len(df)} records")
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-informed derived features."""
    df = df.copy()

    # Vulnerability score (higher = more vulnerable)
    df["vulnerability_score"] = (
        (df["age"] >= 60).astype(int) * 2
        + df["is_widow"].astype(int) * 2
        + df["is_disabled"].astype(int) * 3
        + (df["disability_percentage"] >= 80).astype(int) * 2
        + df["primary_earner_deceased"].astype(int) * 2
        + (df["annual_income"] < 30000).astype(int)
        + (df["household_size"] > 5).astype(int)
    )

    # Age group bins
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 17, 29, 44, 59, 79, 120],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(int)

    # Income per capita
    df["income_per_capita"] = (df["annual_income"] / df["household_size"].clip(1)).round(0)

    # Binary flags
    df["is_senior"]          = (df["age"] >= 60).astype(int)
    df["is_severely_disabled"] = ((df["is_disabled"] == 1) & (df["disability_percentage"] >= 80)).astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn ColumnTransformer for feature transformation."""
    cat_cols_extended = CATEGORICAL_COLS  # state + district + ration + edu

    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline,
             NUMERICAL_COLS + ["vulnerability_score", "age_group",
                               "income_per_capita", "is_senior",
                               "is_severely_disabled"]),
            ("cat", categorical_pipeline, cat_cols_extended),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_splits(df: pd.DataFrame,
                   test_size: float = 0.2,
                   val_size: float = 0.1,
                   random_state: int = 42):
    """Split dataset into train / validation / test sets."""
    df = clean(df)
    df = feature_engineer(df)

    # Drop non-feature columns
    drop = [c for c in DROP_COLS if c in df.columns]
    X    = df.drop(columns=[TARGET_COL] + drop)
    y    = df[TARGET_COL].astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size / (1 - test_size),
        stratify=y_temp,
        random_state=random_state,
    )

    print(f"[Preprocessing] Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_preprocessor(preprocessor, path: str = "models/preprocessor.pkl"):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"[Preprocessing] Preprocessor saved → {path}")


def load_preprocessor(path: str = "models/preprocessor.pkl"):
    return joblib.load(path)
