"""Plot cross-platform reproducibility results.

Run:
    python reproducibility/plot_reproducibility.py

Input:
    data/results/reproducibility_results.csv
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "results" / "reproducibility_results.csv"
OUT = ROOT / "data" / "results" / "reproducibility_figures"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} does not exist. Record at least two platform runs first."
        )

    df = pd.read_csv(INPUT)
    df["test_accuracy"] = pd.to_numeric(df["test_accuracy"], errors="coerce")

    OUT.mkdir(parents=True, exist_ok=True)

    # Platform-wise accuracy comparison.
    summary = (
        df.groupby(["experiment", "platform"])["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    pivot = summary.pivot(
        index="experiment",
        columns="platform",
        values="mean",
    )

    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("Experiment")
    ax.set_title("Cross-platform reproducibility")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Platform")
    plt.tight_layout()
    plt.savefig(
        OUT / "platform_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Exact-seed differences between platforms.
    platforms = list(df["platform"].dropna().unique())

    if len(platforms) >= 2:
        reference = platforms[0]

        pivot_seed = df.pivot_table(
            index=["experiment", "seed"],
            columns="platform",
            values="test_accuracy",
            aggfunc="mean",
        )

        other_platforms = [p for p in platforms if p != reference]

        for platform in other_platforms:
            if reference not in pivot_seed.columns or platform not in pivot_seed.columns:
                continue

            difference = (
                pivot_seed[platform] - pivot_seed[reference]
            ).dropna()

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(np.arange(len(difference)), difference.values)
            ax.axhline(0, linewidth=1)
            ax.set_xlabel("Experiment / seed")
            ax.set_ylabel(
                f"Accuracy difference: {platform} − {reference} "
                "(percentage points)"
            )
            ax.set_title("Cross-platform accuracy difference")
            ax.set_xticks(range(len(difference)))
            ax.set_xticklabels(
                [f"{idx[0]} / seed {idx[1]}" for idx in difference.index],
                rotation=45,
                ha="right",
            )
            fig.tight_layout()
            fig.savefig(
                OUT / f"accuracy_difference_{platform}_vs_{reference}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    # Numerical summary.
    summary.to_csv(
        OUT / "reproducibility_summary.csv",
        index=False,
    )

    print(f"Saved reproducibility figures to {OUT}")


if __name__ == "__main__":
    main()
