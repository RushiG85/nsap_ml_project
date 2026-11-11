"""
EDA & Evaluation Visualizations
=================================
Generates all plots used in analysis and model evaluation reports.
Saves figures to reports/ directory.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

SCHEME_NAMES  = {0: "IGNOAPS", 1: "IGNWPS", 2: "IGNDPS", 3: "NFBS"}
SCHEME_COLORS = ["#3b82d4", "#7c5cd8", "#e05050", "#22a87a"]
# Resolve relative to this file so plots are always written to the correct
# project-level reports/ directory regardless of the working directory.
REPORT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "reports")


def ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


# ── 1. Class distribution ────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame):
    ensure_report_dir()
    counts = df["scheme_name"].value_counts().reindex(
        [SCHEME_NAMES[i] for i in sorted(SCHEME_NAMES)]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=SCHEME_COLORS, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("NSAP Scheme Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Scheme")
    ax.set_ylabel("Number of Applicants")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plots] Saved → {path}")


# ── 2. Age distribution by scheme ────────────────────────────────────────────
def plot_age_by_scheme(df: pd.DataFrame):
    ensure_report_dir()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    for ax, (sid, sname) in zip(axes.flat, SCHEME_NAMES.items()):
        subset = df[df["scheme"] == sid]["age"]
        ax.hist(subset, bins=20, color=SCHEME_COLORS[sid], edgecolor="white", alpha=0.85)
        ax.set_title(sname, fontsize=12, fontweight="bold")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(subset.median(), color="black", linestyle="--", linewidth=1.2,
                   label=f"Median {subset.median():.0f}")
        ax.legend(fontsize=9)
    fig.suptitle("Age Distribution by NSAP Scheme", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "age_by_scheme.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plots] Saved → {path}")


# ── 3. Correlation heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    ensure_report_dir()
    num_cols = [
        "age", "gender", "is_widow", "is_disabled", "disability_percentage",
        "annual_income", "land_holding_acres", "household_size",
        "primary_earner_deceased", "scheme",
    ]
    corr = df[[c for c in num_cols if c in df.columns]].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plots] Saved → {path}")


# ── 4. Confusion matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(cm: list, model_name: str = "Model"):
    ensure_report_dir()
    cm_arr = np.array(cm)
    labels = [SCHEME_NAMES[i] for i in sorted(SCHEME_NAMES)]

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_arr, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plots] Saved → {path}")


# ── 5. CV comparison bar chart ────────────────────────────────────────────────
def plot_cv_comparison(cv_results: dict):
    ensure_report_dir()
    names  = list(cv_results.keys())
    means  = [cv_results[n]["mean_f1"] for n in names]
    stds   = [cv_results[n]["std_f1"]  for n in names]

    colors = ["#3b82d4" if m != max(means) else "#22a87a" for m in means]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=colors, edgecolor="white", error_kw={"linewidth": 1.5})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{m:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel("Weighted F1 Score (5-Fold CV)")
    ax.set_title("Model Comparison — Cross-Validation F1 Scores", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "cv_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plots] Saved → {path}")


# ── 6. Feature importance ─────────────────────────────────────────────────────
def plot_feature_importance(pipeline, top_n: int = 20):
    ensure_report_dir()
    clf = pipeline.named_steps["classifier"]
    pre = pipeline.named_steps["preprocessor"]

    if not hasattr(clf, "feature_importances_"):
        print("[Plots] Classifier does not expose feature_importances_ – skipping")
        return

    # Get feature names from ColumnTransformer
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(len(clf.feature_importances_))]

    importances  = clf.feature_importances_
    sorted_idx   = np.argsort(importances)[::-1][:top_n]
    top_names    = [feature_names[i].replace("num__", "").replace("cat__", "")
                    for i in sorted_idx]
    top_vals     = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(top_n), top_vals[::-1], color="#3b82d4", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plots] Saved → {path}")


# ── 7. Per-class F1 scores ────────────────────────────────────────────────────
def plot_per_class_f1(report: dict):
    ensure_report_dir()
    labels = [SCHEME_NAMES[i] for i in sorted(SCHEME_NAMES)]
    f1s    = [report[l]["f1-score"] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, f1s, color=SCHEME_COLORS, edgecolor="white")
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score on Test Set", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "per_class_f1.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plots] Saved → {path}")


def generate_all_plots(df: pd.DataFrame, metrics: dict, pipeline):
    """Convenience wrapper – generate all evaluation plots at once."""
    plot_class_distribution(df)
    plot_age_by_scheme(df)
    plot_correlation_heatmap(df)

    if "confusion_matrix" in metrics:
        plot_confusion_matrix(metrics["confusion_matrix"],
                              model_name=metrics.get("model_name", "Best Model"))

    if "cv_results" in metrics:
        plot_cv_comparison(metrics["cv_results"])

    if "classification_report" in metrics:
        plot_per_class_f1(metrics["classification_report"])

    plot_feature_importance(pipeline)
    print("[Plots] All plots generated in reports/")
