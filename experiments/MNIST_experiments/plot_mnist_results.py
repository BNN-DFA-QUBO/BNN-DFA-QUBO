import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "data" / "mnist_controlled_results.csv"
OUTPUT_DIR = ROOT / "data" / "mnist_results"

METRICS = {
    "test_accuracy": "Test Accuracy",
}


def load_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing results file: {RESULTS_PATH}\n"
            "Run: python -m experiments.MNIST_experiments.run_mnist_controlled"
        )
    return pd.read_csv(RESULTS_PATH)


def save_summary(df):
    summary = (
        df.groupby(["experiment", "method"])["test_accuracy"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    path = OUTPUT_DIR / "mnist_results_summary.csv"
    summary.to_csv(path, index=False)
    print(f"Saved: {path}")
    return summary


def plot_mean_std(df):
    summary = (
        df.groupby(["experiment", "method"])["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(11, 6))
    plt.bar(
        summary["method"],
        summary["mean"],
        yerr=summary["std"],
        capsize=5,
    )
    plt.ylabel("Test Accuracy (%)")
    plt.title("MNIST Results — Mean ± Standard Deviation")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 100)
    plt.tight_layout()

    path = OUTPUT_DIR / "test_accuracy_mean_std.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_per_seed(df):
    pivot = df.pivot(
        index="seed",
        columns="method",
        values="test_accuracy",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 6),
    )
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlabel("Seed")
    ax.set_title("MNIST Results Across Seeds")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=0)
    plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    path = OUTPUT_DIR / "test_accuracy_per_seed.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()

    print(f"Loaded {len(df)} experiment results.")
    save_summary(df)
    plot_mean_std(df)
    plot_per_seed(df)

    print("\nMNIST result plots generated successfully.")


if __name__ == "__main__":
    main()
