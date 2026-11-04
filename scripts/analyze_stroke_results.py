#!/usr/bin/env python3
"""Reproducible analysis of stroke-prediction classifier-head experiments.

Run from the repository root:
    python scripts/analyze_stroke_results.py

Inputs:
    results/stroke_oversampling_results.csv
    results/stroke_no_balancing_results.csv
    results/stroke_performance_summary.csv

Outputs are written to results/analysis/.
The script summarizes the existing runs; it does not retrain models.
"""
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "analysis"

FILES = {
    "oversampling": RESULTS / "stroke_oversampling_results.csv",
    "no balancing": RESULTS / "stroke_no_balancing_results.csv",
}
METRICS = ["balanced_accuracy", "f1", "roc_auc", "pr_auc"]
DISPLAY = {
    "balanced_accuracy": "Balanced accuracy (%)",
    "f1": "F1 (%)",
    "roc_auc": "ROC-AUC (%)",
    "pr_auc": "PR-AUC (%)",
}
EXPECTED_HEADS = ["Real LS", "Binarized LS", "Direct Binary", "STE Binary", "QUBO Binary"]


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def load_and_validate():
    frames = []
    required = {"method", "seed", *METRICS}
    for condition, path in FILES.items():
        if not path.exists():
            fail(f"Missing input file: {path}\nPut the CSV in the repository's results/ folder.")
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            fail(f"{path.name} is missing required columns: {sorted(missing)}")
        if len(df) != 25:
            warnings.warn(f"{path.name} has {len(df)} rows; expected 25 (5 heads x 5 seeds).")
        if df.duplicated(["method", "seed"]).any():
            fail(f"{path.name} contains duplicate method/seed rows.")
        if set(df["method"].dropna()) != set(EXPECTED_HEADS):
            fail(f"Unexpected classifier-head names in {path.name}: {sorted(df['method'].dropna().unique())}")
        if df[METRICS].isna().any().any():
            fail(f"Missing metric values found in {path.name}.")
        df = df.copy()
        df["balancing_method"] = condition
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def calculate_summary(df):
    rows = []
    for (condition, head), group in df.groupby(["balancing_method", "method"], sort=False):
        row = {"balancing_method": condition, "classifier_head": head, "n_runs": len(group)}
        for metric in METRICS:
            # CSV metrics are fractions; report mean and sample SD as percentages.
            values = group[metric].astype(float) * 100
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
        rows.append(row)
    order = {name: i for i, name in enumerate(EXPECTED_HEADS)}
    summary = pd.DataFrame(rows)
    summary["_order"] = summary["classifier_head"].map(order)
    summary["_condition_order"] = summary["balancing_method"].map({"oversampling": 0, "no balancing": 1})
    return summary.sort_values(["_condition_order", "_order"]).drop(columns=["_order", "_condition_order"]).reset_index(drop=True)


def compare_published_summary(summary):
    path = RESULTS / "stroke_performance_summary.csv"
    if not path.exists():
        print("NOTE: published summary CSV not found; skipped cross-check.")
        return
    published = pd.read_csv(path)
    # Support both the current schema and the previous schema.
    if {"balancing_method", "classifier_head"}.issubset(published.columns):
        key_cols = ["balancing_method", "classifier_head"]
        metric_cols = [f"{m}_{stat}" for m in METRICS for stat in ("mean", "std")]
        check = summary.merge(published, on=key_cols, suffixes=("_calculated", "_published"))
        if len(check) != len(summary):
            fail("Published summary does not contain the same condition/head combinations.")
        bad = []
        for col in metric_cols:
            a = check[f"{col}_calculated"].to_numpy(float)
            b = check[f"{col}_published"].to_numpy(float)
            if not np.allclose(a, b, rtol=1e-9, atol=1e-8):
                bad.append(col)
        if bad:
            fail(f"Published summary differs from raw runs for: {bad}")
        print("PASS: published summary agrees with calculations from individual runs.")
    else:
        print("NOTE: summary uses an older schema; calculated a fresh analysis from raw runs.")


def bar_chart(summary, metric, filename, error_bars=True):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(EXPECTED_HEADS))
    width = 0.36
    for i, condition in enumerate(["oversampling", "no balancing"]):
        subset = summary[summary["balancing_method"] == condition].set_index("classifier_head").reindex(EXPECTED_HEADS)
        means = subset[f"{metric}_mean"].to_numpy()
        stds = subset[f"{metric}_std"].to_numpy()
        ax.bar(x + (i - 0.5) * width, means, width, label=condition.title(),
               yerr=stds if error_bars else None, capsize=3 if error_bars else 0)
    ax.set_xticks(x)
    ax.set_xticklabels(EXPECTED_HEADS, rotation=15, ha="right")
    ax.set_ylabel(DISPLAY[metric])
    ax.set_title(f"{DISPLAY[metric]} by classifier head and balancing condition")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200)
    plt.close(fig)


def balancing_effect(summary, metric):
    pivot = summary.pivot(index="classifier_head", columns="balancing_method", values=f"{metric}_mean")
    result = pd.DataFrame({
        "classifier_head": pivot.index,
        "oversampling_mean_pct": pivot["oversampling"].values,
        "no_balancing_mean_pct": pivot["no balancing"].values,
    })
    result["oversampling_minus_no_balancing_pp"] = (
        result["oversampling_mean_pct"] - result["no_balancing_mean_pct"]
    )
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_and_validate()
    summary = calculate_summary(df)
    compare_published_summary(summary)

    # Save a freshly calculated summary, including run counts.
    summary.to_csv(OUT / "recalculated_performance_summary.csv", index=False)

    # Four headline metric plots; error bars are sample SD across the five runs.
    for metric in METRICS:
        bar_chart(summary, metric, f"{metric}_comparison.png", error_bars=True)

    # Focused comparison of balancing effect for each metric.
    effects = []
    for metric in METRICS:
        effect = balancing_effect(summary, metric)
        effect.insert(0, "metric", metric)
        effects.append(effect)
    effects_df = pd.concat(effects, ignore_index=True)
    effects_df.to_csv(OUT / "balancing_effects.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(EXPECTED_HEADS))
    width = 0.18
    for i, metric in enumerate(METRICS):
        effect = balancing_effect(summary, metric).set_index("classifier_head").reindex(EXPECTED_HEADS)
        ax.bar(x + (i - 1.5) * width, effect["oversampling_minus_no_balancing_pp"].values,
               width, label=metric.replace("_", " ").upper())
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(EXPECTED_HEADS, rotation=15, ha="right")
    ax.set_ylabel("Oversampling minus no balancing (percentage points)")
    ax.set_title("Effect of oversampling by classifier head")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "oversampling_effect.png", dpi=200)
    plt.close(fig)

    # Run-level diagnostics, including potential all-positive predictions.
    diag_cols = [c for c in ["balancing_method", "method", "seed", "balanced_accuracy",
                             "precision", "recall", "f1", "roc_auc", "pr_auc",
                             "predicted_positive", "actual_positive", "threshold",
                             "bias", "validation_balanced_accuracy"] if c in df.columns]
    df[diag_cols].to_csv(OUT / "run_level_diagnostics.csv", index=False)
    if "predicted_positive" in df.columns and "actual_positive" in df.columns:
        suspicious = df[
            (df["predicted_positive"] == df["actual_positive"]) &
            (df["actual_positive"] > 0)
        ]
        if not suspicious.empty:
            print("\nWARNING: runs predict exactly as many positives as the test set contains.")
            print("Inspect run_level_diagnostics.csv; this is a diagnostic flag, not proof of a bug.")
            print(suspicious[["balancing_method", "method", "seed", "predicted_positive",
                              "actual_positive", "threshold", "bias"]].to_string(index=False))

    print("\nAnalysis complete.")
    print(f"Rows analyzed: {len(df)} (expected 50 total runs).")
    print(f"Classifier heads: {', '.join(EXPECTED_HEADS)}")
    print(f"Outputs: {OUT}")
    print("\nKey mean results (percent; sorted by condition/head):")
    show_cols = ["balancing_method", "classifier_head"] + [f"{m}_mean" for m in METRICS]
    print(summary[show_cols].round(2).to_string(index=False))
    print("\nNote: error bars describe variation across training seeds on the same test split;")
    print("they are not dataset-level confidence intervals. The test set has only 38 positives.")


if __name__ == "__main__":
    main()
