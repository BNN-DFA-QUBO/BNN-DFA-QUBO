"""Create paper-ready MNIST result tables from data/results/mnist_results.csv.

Run:
    python experiments/MNIST_experiments/collect_results.py

This script does not train anything.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "results" / "mnist_results.csv"
OUT = ROOT / "data" / "results"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} does not exist. Run the experiments with result logging first."
        )

    df = pd.read_csv(INPUT)

    # Convert numeric columns where possible.
    for col in [
        "test_accuracy",
        "test_loss",
        "training_time_sec",
        "parameter_count",
        "qubo_time_sec",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    tables = build_aggregate_tables(df)
    paths = [OUT / name for name in (
        "main_classification_results.csv", "computational_cost_results.csv",
        "platform_reproducibility_results.csv",
    ) if name != "platform_reproducibility_results.csv" or "platform" in df.columns]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to replace aggregate result tables: " + ", ".join(existing))
    for name, table in tables.items():
        table.to_csv(OUT / name, index=False, mode="x")

    print(f"Saved result tables in {OUT}")


def build_aggregate_tables(df):
    """Build summaries without pooling incompatible result protocols/configurations."""
    protocol_keys = [key for key in (
        "experiment_type", "qubo_grid_protocol", "qubo_num_reads_selected",
        "qubo_num_sweeps_selected", "qubo_scale_multiplier_selected",
        "qubo_selection_metric", "validation_fraction", "preprocessing_version",
        "split_seed", "balancing_method",
    ) if key in df.columns]
    group_keys = [key for key in ("experiment", "method", *protocol_keys) if key in df.columns]
    if not group_keys:
        raise ValueError("Results must include at least one experiment/method grouping column")

    # Main results: mean ± std over compatible protocol/configuration groups.
    main = (
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
    # Computational-cost summary.
    cost = (
        df.groupby(group_keys, dropna=False)
        .agg(
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_std=("test_accuracy", "std"),
            training_time_mean_sec=("training_time_sec", "mean"),
            parameter_count=("parameter_count", "first"),
            qubo_time_mean_sec=("qubo_time_sec", "mean"),
        )
        .reset_index()
    )
    tables = {
        "main_classification_results.csv": main,
        "computational_cost_results.csv": cost,
    }
    # Platform-specific reproducibility summary.
    if "platform" in df.columns:
        repro = (
            df.groupby([*group_keys, "platform"], dropna=False)
            ["test_accuracy"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )
        tables["platform_reproducibility_results.csv"] = repro
    return tables


if __name__ == "__main__":
    main()
