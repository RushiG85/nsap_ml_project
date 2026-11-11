"""
Prediction & Inference Module
==============================
Loads the trained pipeline and predicts the correct NSAP scheme
for new applicant records. Works both locally and via IBM WML endpoint.
"""

import json
import os
import joblib
import pandas as pd
import numpy as np

SCHEME_NAMES = {0: "IGNOAPS", 1: "IGNWPS", 2: "IGNDPS", 3: "NFBS"}
SCHEME_DESC  = {
    0: "Indira Gandhi National Old Age Pension Scheme",
    1: "Indira Gandhi National Widow Pension Scheme",
    2: "Indira Gandhi National Disability Pension Scheme",
    3: "National Family Benefit Scheme",
}
ELIGIBILITY_RULES = {
    0: "Age ≥ 60, BPL household",
    1: "Female, Widow, Age 18–79, BPL household",
    2: "Disability ≥ 80%, Age 18–79, BPL household",
    3: "Primary earner deceased, Age 18–64, BPL household",
}


# ── Local inference ───────────────────────────────────────────────────────────
class NSAPPredictor:
    """
    Wraps a trained sklearn pipeline for NSAP scheme prediction.

    Usage
    -----
    predictor = NSAPPredictor.load("models/best_model.pkl")
    result    = predictor.predict_one({
        "state": "Bihar", "district": "Biha_D1",
        "age": 65, "gender": 1, "is_widow": 1, "is_disabled": 0,
        "disability_percentage": 0, "bpl_card": 1,
        "primary_earner_deceased": 0, "annual_income": 25000,
        "land_holding_acres": 0.5, "household_size": 4,
        "education_level": 0, "ration_card_type": 0,
    })
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def load(cls, path: str = "models/best_model.pkl"):
        pipeline = joblib.load(path)
        print(f"[Predictor] Model loaded from {path}")
        return cls(pipeline)

    # -- feature engineering mirror (must match preprocessing.py) -------------
    @staticmethod
    def _engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["vulnerability_score"] = (
            (df["age"] >= 60).astype(int) * 2
            + df["is_widow"].astype(int) * 2
            + df["is_disabled"].astype(int) * 3
            + (df["disability_percentage"] >= 80).astype(int) * 2
            + df["primary_earner_deceased"].astype(int) * 2
            + (df["annual_income"] < 30000).astype(int)
            + (df["household_size"] > 5).astype(int)
        )
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 17, 29, 44, 59, 79, 120],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype(int)
        df["income_per_capita"]     = (df["annual_income"] / df["household_size"].clip(1)).round(0)
        df["is_senior"]             = (df["age"] >= 60).astype(int)
        df["is_severely_disabled"]  = (
            (df["is_disabled"] == 1) & (df["disability_percentage"] >= 80)
        ).astype(int)
        return df

    def predict_one(self, applicant: dict) -> dict:
        """Predict the NSAP scheme for a single applicant record."""
        df  = pd.DataFrame([applicant])
        df  = self._engineer(df)
        sid = int(self.pipeline.predict(df)[0])

        proba = None
        if hasattr(self.pipeline.named_steps["classifier"], "predict_proba"):
            proba_arr = self.pipeline.predict_proba(df)[0]
            proba = {SCHEME_NAMES[i]: round(float(p), 4)
                     for i, p in enumerate(proba_arr)}

        return {
            "applicant"              : applicant,
            "predicted_scheme_id"    : sid,
            "predicted_scheme_name"  : SCHEME_NAMES[sid],
            "scheme_description"     : SCHEME_DESC[sid],
            "eligibility_criteria"   : ELIGIBILITY_RULES[sid],
            "probabilities"          : proba,
            "confidence"             : round(float(max(proba.values())), 4) if proba else None,
        }

    def predict_batch(self, applicants: list) -> list:
        """Predict schemes for a list of applicant dicts."""
        df  = pd.DataFrame(applicants)
        df  = self._engineer(df)
        sid = self.pipeline.predict(df).astype(int)

        proba_all = None
        if hasattr(self.pipeline.named_steps["classifier"], "predict_proba"):
            proba_all = self.pipeline.predict_proba(df)

        results = []
        for i, s in enumerate(sid):
            proba = None
            if proba_all is not None:
                proba = {SCHEME_NAMES[j]: round(float(p), 4)
                         for j, p in enumerate(proba_all[i])}
            results.append({
                "predicted_scheme_id"   : int(s),
                "predicted_scheme_name" : SCHEME_NAMES[int(s)],
                "probabilities"         : proba,
                "confidence"            : round(float(max(proba.values())), 4) if proba else None,
            })
        return results


# ── IBM WML remote inference ──────────────────────────────────────────────────
def predict_via_wml(applicants: list,
                    deployment_info_path: str = "models/deployment_info.json",
                    creds_file: str = "ibm_credentials.json") -> list:
    """
    Send applicant records to the deployed IBM WML endpoint for scoring.

    Requires:
      - models/deployment_info.json (written by ibm_wml_deploy.py)
      - ibm_credentials.json with valid API key
    """
    from ibm_wml_deploy import load_credentials, get_wml_client

    with open(deployment_info_path) as f:
        dep_info = json.load(f)

    creds  = load_credentials(creds_file)
    client = get_wml_client(creds)

    from ibm_wml_deploy import score_records
    return score_records(client, dep_info["deployment_id"], applicants)


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "../models/best_model.pkl"

    predictor = NSAPPredictor.load(model_path)

    test_cases = [
        {   # Should → IGNOAPS (old age)
            "state": "Bihar", "district": "Biha_D2",
            "age": 68, "gender": 0, "is_widow": 0, "is_disabled": 0,
            "disability_percentage": 0, "bpl_card": 1,
            "primary_earner_deceased": 0, "annual_income": 22000,
            "land_holding_acres": 0.3, "household_size": 3,
            "education_level": 0, "ration_card_type": 0,
        },
        {   # Should → IGNWPS (widow pension)
            "state": "Uttar Pradesh", "district": "Uttar_D1",
            "age": 45, "gender": 1, "is_widow": 1, "is_disabled": 0,
            "disability_percentage": 0, "bpl_card": 1,
            "primary_earner_deceased": 1, "annual_income": 18000,
            "land_holding_acres": 0.0, "household_size": 5,
            "education_level": 0, "ration_card_type": 1,
        },
        {   # Should → IGNDPS (disability)
            "state": "Tamil Nadu", "district": "Tami_D3",
            "age": 35, "gender": 0, "is_widow": 0, "is_disabled": 1,
            "disability_percentage": 90, "bpl_card": 1,
            "primary_earner_deceased": 0, "annual_income": 15000,
            "land_holding_acres": 0.1, "household_size": 4,
            "education_level": 1, "ration_card_type": 0,
        },
        {   # Should → NFBS (family benefit)
            "state": "Odisha", "district": "Odis_D1",
            "age": 30, "gender": 1, "is_widow": 0, "is_disabled": 0,
            "disability_percentage": 0, "bpl_card": 1,
            "primary_earner_deceased": 1, "annual_income": 20000,
            "land_holding_acres": 0.5, "household_size": 6,
            "education_level": 0, "ration_card_type": 0,
        },
    ]

    print("\n" + "=" * 60)
    print("  NSAP SCHEME PREDICTION — DEMO CASES")
    print("=" * 60)
    for i, case in enumerate(test_cases, 1):
        result = predictor.predict_one(case)
        print(f"\nApplicant #{i}")
        print(f"  Age={case['age']}, Gender={'F' if case['gender']==1 else 'M'}, "
              f"Widow={bool(case['is_widow'])}, Disabled={bool(case['is_disabled'])}")
        print(f"  ➤ Predicted: {result['predicted_scheme_name']} "
              f"({result['scheme_description']})")
        print(f"  ➤ Confidence: {result['confidence']:.2%}")
        if result["probabilities"]:
            for sname, p in result["probabilities"].items():
                bar = "█" * int(p * 20)
                print(f"     {sname:10s} {p:.4f}  {bar}")
