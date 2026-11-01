"""Generate paper-ready figures for the CIFAR-10 experiments.

Saves figures into data/results/cifar10_figures/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "results" / "cifar10_results.csv"
OUT = ROOT / "data" / "results" / "cifar10_figures"


def save_figure(fig: plt.Figure, name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {target}")


def plot_main_results(df: pd.DataFrame, out_dir: Path):
    if "representation_path" in df.columns:
        df = df.copy()
        df["epochs"] = df["representation_path"].astype(str).str.extract(r"epochs_(\d+)").fillna(20).astype(int)
        max_epochs = df["epochs"].max()
        df = df[df["epochs"] == max_epochs]

    summary = (
        df.groupby("method")["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        summary["method"],
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=5,
        color="#3470a3",
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("CIFAR-10 BNN + DFA: Classifier Head Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=20, ha="right", fontsize=10)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    save_figure(fig, "fig1_cifar10_main_classification_accuracy.png", out_dir)


def plot_computational_cost(df: pd.DataFrame, out_dir: Path):
    if "head_fit_time_sec" not in df.columns:
        return
    if "representation_path" in df.columns:
        df = df.copy()
        df["epochs"] = df["representation_path"].astype(str).str.extract(r"epochs_(\d+)").fillna(20).astype(int)
        max_epochs = df["epochs"].max()
        df = df[df["epochs"] == max_epochs]

    summary = (
        df.groupby("method")["head_fit_time_sec"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        summary["method"],
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=5,
        color="#d95f02",
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_ylabel("Head Optimization Time (seconds)", fontsize=12)
    ax.set_title("CIFAR-10 Head Optimization Computational Cost", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=20, ha="right", fontsize=10)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}s",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    save_figure(fig, "fig2_cifar10_computational_cost.png", out_dir)


def main():
    parser = argparse.ArgumentParser(description="Plot CIFAR-10 experimental results")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} does not exist.")

    df = pd.read_csv(args.input)
    for col in ["test_accuracy", "head_fit_time_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    plot_main_results(df, args.out_dir)
    plot_computational_cost(df, args.out_dir)


if __name__ == "__main__":
    main()
