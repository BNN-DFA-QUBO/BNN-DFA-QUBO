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

    # Main results: mean ± std over seeds when multiple seeds exist.
    main = (
        df.groupby(["experiment", "method"], dropna=False)["test_accuracy"]
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
    main.to_csv(OUT / "main_classification_results.csv", index=False)

    # Computational-cost summary.
    cost = (
        df.groupby(["experiment", "method"], dropna=False)
        .agg(
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_std=("test_accuracy", "std"),
            training_time_mean_sec=("training_time_sec", "mean"),
            parameter_count=("parameter_count", "first"),
            qubo_time_mean_sec=("qubo_time_sec", "mean"),
        )
        .reset_index()
    )
    cost.to_csv(OUT / "computational_cost_results.csv", index=False)

    # Platform-specific reproducibility summary.
    if "platform" in df.columns:
        repro = (
            df.groupby(["experiment", "method", "platform"], dropna=False)
            ["test_accuracy"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )
        repro.to_csv(OUT / "platform_reproducibility_results.csv", index=False)

    print(f"Saved result tables in {OUT}")


if __name__ == "__main__":
    main()
