"""
NSAP Scheme Classifier — Main Pipeline Orchestrator
=====================================================
Run this script to execute the full end-to-end pipeline:
  1. Generate / load dataset
  2. Preprocess and split
  3. Train + evaluate models
  4. Generate visualizations

Usage
-----
  python main.py              # full training pipeline
  python main.py --predict    # also run demo predictions after training
"""

import argparse
import os
import sys

# Add src/ to path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_generator  import load_or_generate
from preprocessing   import prepare_splits, build_preprocessor, save_preprocessor
from train           import train_and_select
from visualize       import generate_all_plots


DATA_PATH   = "data/nsap_dataset.csv"
MODEL_DIR   = "models"
REPORT_DIR  = "reports"


def run_pipeline(predict_demo: bool = False):
    print("\n" + "=" * 65)
    print("  NSAP ELIGIBILITY PREDICTION — ML PIPELINE")
    print("=" * 65)

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading / Generating Dataset …")
    df = load_or_generate(path=DATA_PATH, n_samples=12000)
    print(f"      Records: {len(df):,}  |  Columns: {df.shape[1]}")
    print(df["scheme_name"].value_counts().to_string())

    # ── Step 2: Preprocessing ─────────────────────────────────────────────────
    print("\n[2/4] Preprocessing & Feature Engineering …")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(df)

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)     # fit ONLY on training data
    save_preprocessor(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

    # ── Step 3: Training ──────────────────────────────────────────────────────
    print("\n[3/4] Training & Selecting Best Model …")
    best_pipeline, metrics = train_and_select(
        preprocessor, X_train, X_val, X_test, y_train, y_val, y_test,
        model_dir=MODEL_DIR,
    )
    print(f"\n  ✔ Best Model  : {metrics['model_name']}")
    print(f"  ✔ Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  ✔ F1 Weighted : {metrics['f1_weighted']:.4f}")
    if metrics.get("roc_auc_weighted"):
        print(f"  ✔ ROC-AUC     : {metrics['roc_auc_weighted']:.4f}")

    # ── Step 4: Visualizations ────────────────────────────────────────────────
    print("\n[4/4] Generating Evaluation Plots …")
    generate_all_plots(df, metrics, best_pipeline)

    # ── Demo predictions ──────────────────────────────────────────────────────
    if predict_demo:
        print("\n── Demo Predictions ────────────────────────────────────────")
        from predict import NSAPPredictor
        predictor = NSAPPredictor.load(os.path.join(MODEL_DIR, "best_model.pkl"))
        demo_cases = [
            {"state": "Bihar", "district": "Biha_D2", "age": 68, "gender": 0,
             "is_widow": 0, "is_disabled": 0, "disability_percentage": 0,
             "bpl_card": 1, "primary_earner_deceased": 0, "annual_income": 22000,
             "land_holding_acres": 0.3, "household_size": 3,
             "education_level": 0, "ration_card_type": 0},
            {"state": "Uttar Pradesh", "district": "Uttar_D1", "age": 45, "gender": 1,
             "is_widow": 1, "is_disabled": 0, "disability_percentage": 0,
             "bpl_card": 1, "primary_earner_deceased": 1, "annual_income": 18000,
             "land_holding_acres": 0.0, "household_size": 5,
             "education_level": 0, "ration_card_type": 1},
        ]
        for i, case in enumerate(demo_cases, 1):
            r = predictor.predict_one(case)
            print(f"  Applicant #{i}: Age={case['age']}, "
                  f"Gender={'F' if case['gender']==1 else 'M'}")
            print(f"    → {r['predicted_scheme_name']} "
                  f"(confidence: {r['confidence']:.1%})")

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE ✔")
    print("=" * 65)
    print(f"  Model   : {MODEL_DIR}/best_model.pkl")
    print(f"  Metrics : {MODEL_DIR}/metrics.json")
    print(f"  Plots   : {REPORT_DIR}/")
    return best_pipeline, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSAP Scheme Classifier Pipeline")
    parser.add_argument("--predict", action="store_true",
                        help="Run demo predictions after training")
    args = parser.parse_args()

    run_pipeline(predict_demo=args.predict)
