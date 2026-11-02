"""Collect and summarize Stroke experimental results with duplicate disambiguation.

Reads:
  - data/results/stroke_results.csv (independent and shared results)
  - data/results/stroke_shared_results.csv (canonical shared-backbone results)

Rules:
  - Preserves historical CSV logs.
  - Distinguishes repeated runs by unique `run_id`.
  - In aggregate summaries, selects the latest execution per (experiment, method, seed)
    so multiple runs on the same seed are not counted as independent observations.
  - Conflicting duplicate runs are reported explicitly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STROKE_RESULTS_CSV = ROOT / "data" / "results" / "stroke_results.csv"
STROKE_SHARED_CSV = ROOT / "data" / "results" / "stroke_shared_results.csv"
OUTPUT_SUMMARY_CSV = ROOT / "data" / "results" / "stroke_summary.csv"


def deduplicate_latest_runs(df: pd.DataFrame, group_cols: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group by method/seed and select the latest entry, recording duplicate diagnostics."""
    if not all(col in df.columns for col in group_cols):
        raise ValueError(f"DataFrame missing required group columns: {group_cols}")

    # Check for duplicates
    dup_mask = df.duplicated(subset=group_cols, keep=False)
    duplicate_rows = df[dup_mask].copy()

    # Sort or keep latest (last row appended in CSV represents latest run)
    deduped = df.drop_duplicates(subset=group_cols, keep="last").copy()
    return deduped, duplicate_rows


def build_stroke_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build mean ± std summary across seeds for each method."""
    numeric_cols = [
        "balanced_accuracy",
        "validation_balanced_accuracy",
        "validation_selection_score",
        "f1",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "head_fit_time_sec",
        "current_run_total_time_sec",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    group_keys = [k for k in ["experiment", "method"] if k in df.columns]
    if "experiment" not in df.columns and "experiment_type" in df.columns:
        group_keys = ["experiment_type", "method"]

    summary = (
        df.groupby(group_keys, dropna=False)["balanced_accuracy"]
        .agg(runs="count", ba_mean="mean", ba_std="std", ba_min="min", ba_max="max")
        .reset_index()
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=STROKE_RESULTS_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_SUMMARY_CSV)
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} total historical rows from {args.input_csv}")

    deduped, dups = deduplicate_latest_runs(df, ["experiment", "method", "seed"])
    if len(dups) > 0:
        print(f"\n[NOTE] Found {len(dups)} duplicate entries across seeds. Disambiguating to latest run:")
        for idx, row in dups.iterrows():
            print(f"  - run_id: {row.get('run_id')} | method: {row.get('method')} | seed: {row.get('seed')} | BA: {row.get('balanced_accuracy')}")
    else:
        print("\nNo duplicate runs found.")

    print(f"\nDeduplicated dataset contains {len(deduped)} observations.")
    summary = build_stroke_summary(deduped)
    print("\nStroke Method Summary (Balanced Accuracy across seeds):")
    print(summary.to_string(index=False))

    summary.to_csv(args.output_csv, index=False)
    print(f"\nSaved summary to: {args.output_csv}")


if __name__ == "__main__":
    main()

