"""
Model Training — Multi-Class NSAP Scheme Classification
=========================================================
Trains multiple classifiers and selects the best via cross-validation.
Models evaluated:
  1. Random Forest
  2. XGBoost
  3. LightGBM
  4. Logistic Regression (baseline)
  5. Gradient Boosting (sklearn)
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

SCHEME_NAMES = {0: "IGNOAPS", 1: "IGNWPS", 2: "IGNDPS", 3: "NFBS"}


def get_candidate_models() -> dict:
    """Return a dictionary of candidate classifiers."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs", C=1.0, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42, n_jobs=-1
        )

    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1,
            verbose=-1
        )

    return models


def build_full_pipeline(preprocessor, classifier) -> Pipeline:
    """Combine preprocessor and classifier into a single sklearn Pipeline."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def cross_validate_models(models: dict, preprocessor,
                           X_train, y_train, cv: int = 5) -> dict:
    """Run stratified k-fold CV for each model and return scores."""
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for name, clf in models.items():
        print(f"  [CV] {name} …", end=" ", flush=True)
        pipe = build_full_pipeline(preprocessor, clf)
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv_splitter, scoring="f1_weighted", n_jobs=-1
        )
        results[name] = {
            "mean_f1": round(float(scores.mean()), 4),
            "std_f1":  round(float(scores.std()), 4),
            "scores":  [round(float(s), 4) for s in scores],
        }
        print(f"F1={scores.mean():.4f} ± {scores.std():.4f}")

    return results


def select_best_model(cv_results: dict) -> str:
    """Pick the model with highest mean CV F1 score."""
    best = max(cv_results, key=lambda k: cv_results[k]["mean_f1"])
    print(f"\n[Training] Best model: {best}  (F1={cv_results[best]['mean_f1']:.4f})")
    return best


def evaluate_on_test(pipeline, X_test, y_test) -> dict:
    """Full evaluation on held-out test set."""
    y_pred = pipeline.predict(X_test)
    y_prob = (pipeline.predict_proba(X_test)
              if hasattr(pipeline.named_steps["classifier"], "predict_proba")
              else None)

    acc    = accuracy_score(y_test, y_pred)
    f1_w   = f1_score(y_test, y_pred, average="weighted")
    f1_m   = f1_score(y_test, y_pred, average="macro")
    cm     = confusion_matrix(y_test, y_pred)

    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(
                y_test, y_prob, multi_class="ovr", average="weighted"
            )
        except Exception:
            pass

    report = classification_report(
        y_test, y_pred,
        target_names=[SCHEME_NAMES[i] for i in sorted(SCHEME_NAMES)],
        output_dict=True,
    )

    metrics = {
        "accuracy"          : round(acc,   4),
        "f1_weighted"       : round(f1_w,  4),
        "f1_macro"          : round(f1_m,  4),
        "roc_auc_weighted"  : round(roc_auc, 4) if roc_auc else None,
        "confusion_matrix"  : cm.tolist(),
        "classification_report": report,
    }

    print(f"\n[Evaluation] Accuracy={acc:.4f}  F1(weighted)={f1_w:.4f}")
    if roc_auc:
        print(f"             ROC-AUC(weighted OvR)={roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[SCHEME_NAMES[i] for i in sorted(SCHEME_NAMES)]
    ))

    return metrics


def train_and_select(preprocessor, X_train, X_val, X_test,
                     y_train, y_val, y_test,
                     model_dir: str = "models") -> tuple:
    """
    Full training workflow:
      1. Cross-validate all candidate models
      2. Select best model
      3. Retrain on train+val
      4. Final evaluation on test set
      5. Save model artefacts
    """
    os.makedirs(model_dir, exist_ok=True)
    models = get_candidate_models()

    print("\n─── Cross-Validation ───────────────────────────────────────")
    cv_results = cross_validate_models(models, preprocessor, X_train, y_train)

    best_name = select_best_model(cv_results)
    best_clf  = models[best_name]

    # Retrain on train + val combined
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])

    print(f"\n─── Final Training ({best_name}) ──────────────────────────")
    best_pipeline = build_full_pipeline(preprocessor, best_clf)
    best_pipeline.fit(X_full, y_full)

    print("\n─── Test Set Evaluation ────────────────────────────────────")
    metrics = evaluate_on_test(best_pipeline, X_test, y_test)

    # Persist artefacts
    model_path   = os.path.join(model_dir, "best_model.pkl")
    metrics_path = os.path.join(model_dir, "metrics.json")

    joblib.dump(best_pipeline, model_path)
    print(f"[Training] Model saved → {model_path}")

    metrics["model_name"] = best_name
    metrics["cv_results"] = cv_results
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Training] Metrics saved → {metrics_path}")

    return best_pipeline, metrics


if __name__ == "__main__":
    from data_generator  import load_or_generate
    from preprocessing   import prepare_splits, build_preprocessor, save_preprocessor

    df           = load_or_generate(path="../data/nsap_dataset.csv")
    splits       = prepare_splits(df)
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)          # fit only on train
    save_preprocessor(preprocessor, "../models/preprocessor.pkl")

    best_pipeline, metrics = train_and_select(
        preprocessor, X_train, X_val, X_test, y_train, y_val, y_test,
        model_dir="../models",
    )
