import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "data" / "reproducibility_results.csv"
OUTPUT_DIR = ROOT / "data" / "reproducibility_results"


def load_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing results file: {RESULTS_PATH}\n"
            "Run the reproducibility experiments first."
        )
    return pd.read_csv(RESULTS_PATH)


def save_summary(df):
    summary = (
        df.groupby(["experiment", "method", "platform"])["test_accuracy"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    path = OUTPUT_DIR / "reproducibility_summary.csv"
    summary.to_csv(path, index=False)
    print(f"Saved: {path}")
    return summary


def plot_platform_means(df):
    summary = (
        df.groupby(["method", "platform"])["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    pivot = summary.pivot(index="method", columns="platform", values="mean")
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlabel("Method")
    ax.set_title("MNIST Reproducibility — Mean Test Accuracy by Platform")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Platform")
    plt.tight_layout()

    path = OUTPUT_DIR / "platform_mean_accuracy.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_platform_spread(df):
    # For each experiment/seed, compare platforms against the first
    # available platform for that exact seed. This is descriptive only:
    # it visualizes numerical divergence, not a "winner".
    pivot = df.pivot_table(
        index=["experiment", "method", "seed"],
        columns="platform",
        values="test_accuracy",
        aggfunc="mean",
    )

    platforms = list(pivot.columns)
    if len(platforms) < 2:
        print("Skipping platform spread plot: fewer than two platforms recorded.")
        return

    reference = platforms[0]
    delta = pivot.subtract(pivot[reference], axis=0).drop(columns=[reference])
    delta = delta.reset_index()

    ax = delta.set_index("method").plot(
        kind="box",
        figsize=(12, 6),
    )
    ax.axhline(0, linewidth=1)
    ax.set_ylabel(f"Accuracy difference vs {reference} (percentage points)")
    ax.set_title("MNIST Reproducibility — Cross-Platform Accuracy Differences")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    path = OUTPUT_DIR / "cross_platform_accuracy_difference.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_per_seed(df):
    # One figure per experiment keeps the platform comparison readable.
    for experiment, group in df.groupby("experiment"):
        pivot = group.pivot(
            index="seed",
            columns="platform",
            values="test_accuracy",
        )

        ax = pivot.plot(
            kind="bar",
            figsize=(9, 5),
        )
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_xlabel("Seed")
        ax.set_title(
            f"Reproducibility Across Platforms — {group['method'].iloc[0]}"
        )
        ax.set_ylim(0, 100)
        plt.xticks(rotation=0)
        plt.legend(title="Platform")
        plt.tight_layout()

        path = OUTPUT_DIR / f"{experiment}_per_seed_platform.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()

    print(f"Loaded {len(df)} reproducibility results.")
    save_summary(df)
    plot_platform_means(df)
    plot_platform_spread(df)
    plot_per_seed(df)

    print("\nReproducibility plots generated successfully.")


if __name__ == "__main__":
    main()
