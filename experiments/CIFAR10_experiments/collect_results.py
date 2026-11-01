"""Aggregate paper-ready result tables from data/results/cifar10_results.csv.

Follows the verified MNIST results collection methodology while keeping
CIFAR-10 outputs strictly separate.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "results" / "cifar10_results.csv"
OUT = ROOT / "data" / "results"


def build_aggregate_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Construct summary tables without pooling incompatible configurations."""
    df = df.copy()
    if "epochs" not in df.columns and "representation_path" in df.columns:
        df["epochs"] = df["representation_path"].astype(str).str.extract(r"epochs_(\d+)").fillna(20).astype(int)

    group_keys = [
        col for col in ["experiment", "epochs", "method", "qubo_tuning_enabled", "validation_fraction"]
        if col in df.columns
    ]
    if not group_keys:
        group_keys = ["method"]

    # 1. Main classification performance
    main_summary = (
        df.groupby(group_keys, dropna=False)["test_accuracy"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "runs",
                "mean": "accuracy_mean",
                "std": "accuracy_std",
                "min": "accuracy_min",
                "max": "accuracy_max",
            }
        )
    )

    # 2. Detailed computational cost breakdown
    cost_summary = (
        df.groupby(group_keys, dropna=False)
        .agg(
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_std=("test_accuracy", "std"),
            backbone_time_mean_sec=("backbone_training_time_sec", "mean"),
            extraction_time_mean_sec=("feature_extraction_time_sec", "mean"),
            head_fit_time_mean_sec=("head_fit_time_sec", "mean"),
            total_time_mean_sec=("current_run_total_time_sec", "mean"),
            backbone_parameters=("backbone_parameter_count", "first"),
            head_parameters=("head_parameter_count", "first"),
            total_parameters=("total_parameter_count", "first"),
            qubo_time_mean_sec=("qubo_time_sec", "mean"),
        )
        .reset_index()
    )

    return {
        "cifar10_main_classification_results.csv": main_summary,
        "cifar10_computational_cost_results.csv": cost_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect CIFAR-10 experiment aggregate tables")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} does not exist. Run CIFAR-10 experiments first."
        )

    df = pd.read_csv(args.input)
    numeric_cols = [
        "test_accuracy",
        "validation_accuracy",
        "training_accuracy",
        "training_time_sec",
        "backbone_training_time_sec",
        "feature_extraction_time_sec",
        "head_fit_time_sec",
        "current_run_total_time_sec",
        "end_to_end_training_time_sec",
        "parameter_count",
        "backbone_parameter_count",
        "head_parameter_count",
        "total_parameter_count",
        "qubo_time_sec",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    tables = build_aggregate_tables(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        out_file = args.out_dir / name
        if out_file.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {out_file}. Specify --overwrite.")
        table.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
