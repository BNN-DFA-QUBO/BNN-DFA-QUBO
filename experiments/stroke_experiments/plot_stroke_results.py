import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_PATH = Path(
    "data/stroke_controlled_results.csv"
)

OUTPUT_DIR = Path(
    "data/stroke_results"
)

METRICS = {
    "balanced_accuracy": "Balanced Accuracy",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "f1": "F1 Score"
}


def load_results():
    return pd.read_csv(RESULTS_PATH)


def summarize_results(df):
    summary = (
        df.groupby("method")[
            [
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc"
            ]
        ]
        .agg(["mean", "std"])
    )

    return summary


def plot_metric(df, metric, title):
    summary = (
        df.groupby("method")[metric]
        .agg(["mean", "std"])
    )

    methods = summary.index.tolist()
    means = summary["mean"] * 100
    stds = summary["std"] * 100

    plt.figure(figsize=(9, 6))

    plt.bar(
        methods,
        means,
        yerr=stds,
        capsize=5
    )

    plt.ylabel("Percentage (%)")
    plt.title(
        f"{title} — Mean ± Standard Deviation"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{metric}_mean_std.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def save_summary_csv(summary):
    output_path = (
        OUTPUT_DIR
        / "stroke_results_summary.csv"
    )

    summary.to_csv(output_path)

    print(
        f"Saved: {output_path}"
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_results()

    print(
        f"Loaded {len(df)} experiments."
    )

    summary = summarize_results(df)

    save_summary_csv(summary)

    for metric, title in METRICS.items():
        plot_metric(
            df,
            metric,
            title
        )

    print()
    print(
        "Stroke result plots generated successfully."
    )


if __name__ == "__main__":
    main()