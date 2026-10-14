"""Summarize and plot results from a controlled Stroke experiment CSV.

Example:
    python -m experiments.stroke_experiments.plot_stroke_results
    python -m experiments.stroke_experiments.plot_stroke_results --input data/stroke_controlled_results.csv --output-dir data/stroke_results/run_1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "stroke_controlled_results.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "stroke_results"

METRICS = {
    "balanced_accuracy": "Balanced Accuracy",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "f1": "F1 Score",
}


def resolve_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input result CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"method", *METRICS.keys(), "precision", "recall"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Result CSV is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"Input result CSV has no rows: {path}")
    return df


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    missing = sorted(set(metrics) - set(df.columns))
    if missing:
        raise ValueError(f"Cannot summarize metrics missing from results: {missing}")
    summary = (
        df.groupby("method", dropna=False)[metrics]
        .agg(["count", "mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def plot_metric(df: pd.DataFrame, metric: str, title: str, output_dir: Path) -> Path:
    summary = df.groupby("method", dropna=False)[metric].agg(["mean", "std", "count"])
    methods = summary.index.astype(str).tolist()
    means = summary["mean"].to_numpy() * 100
    stds = summary["std"].fillna(0).to_numpy() * 100

    fig, ax = plt.subplots(figsize=(max(9, len(methods) * 1.5), 6))
    ax.bar(methods, means, yerr=stds, capsize=5)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"{title} — Mean ± Standard Deviation")
    ax.tick_params(axis="x", labelrotation=20)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()

    output_path = output_dir / f"{metric}_mean_std.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Controlled Stroke result CSV to summarize")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for summary CSV and plots")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace summary/plot files already present in the output directory")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    df = load_results(input_path)

    expected_outputs = [
        output_dir / "stroke_results_summary.csv",
        *(output_dir / f"{metric}_mean_std.png" for metric in METRICS),
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Plot outputs already exist and are protected from replacement: "
            + ", ".join(map(str, existing))
            + ". Choose a new --output-dir or pass --overwrite."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(df)
    summary_path = output_dir / "stroke_results_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Loaded {len(df)} result rows from: {input_path}")
    print(f"Summary saved to: {summary_path}")
    for metric, title in METRICS.items():
        path = plot_metric(df, metric, title, output_dir)
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
