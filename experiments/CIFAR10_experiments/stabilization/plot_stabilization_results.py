"""Generate publication-ready figures for the CIFAR-10 BNN-DFA stabilization study."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAB_RESULTS_DIR = Path("data/cifar10_stabilization/results")
FIG_DIR = Path("data/cifar10_stabilization/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_loss_and_activation_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Baseline data (from diagnostic_phase_a.py)
    epochs_base = [1, 2, 5, 10, 20]
    loss_base = [2.24, 3.28, 12.26, 33.44, 120.46]
    hnorm_base = [53.29, 152.30, 522.65, 1383.45, 4335.30]

    # Stabilized data (from cifar10_representation_progression.csv)
    df_prog = pd.read_csv(STAB_RESULTS_DIR / "cifar10_representation_progression.csv")
    df_mean = df_prog[df_prog["epoch"] > 0].groupby("epoch").mean(numeric_only=True).reset_index()
    epochs_stab = df_mean["epoch"].values
    loss_stab = df_mean["bnn_train_loss"].values
    hnorm_stab = df_mean["H_norm"].values

    # Plot 1: Cross-Entropy Loss
    ax1.plot(epochs_base, loss_base, "o--", color="#d95f02", label="Baseline BNN+DFA (Runaway Explosion)", linewidth=2, markersize=6)
    ax1.plot(epochs_stab, loss_stab, "s-", color="#1b9e77", label="Stabilized BNN+DFA (Controlled)", linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss (log scale)", fontsize=11)
    ax1.set_yscale("log")
    ax1.set_title("Training Loss Dynamics Across Epochs", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Plot 2: Hidden Feature Norm (H_norm)
    ax2.plot(epochs_base, hnorm_base, "o--", color="#d95f02", label="Baseline Representation (Unbounded Scale)", linewidth=2, markersize=6)
    ax2.plot(epochs_stab, hnorm_stab, "s-", color="#1b9e77", label="Stabilized Representation (Bounded Scale)", linewidth=2, markersize=6)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Mean Activation L2 Norm (log scale)", fontsize=11)
    ax2.set_yscale("log")
    ax2.set_title("Feature Representation Norm ($||H||_2$)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    out_path = FIG_DIR / "fig3_loss_and_activation_stability.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_representation_progression():
    df_prog = pd.read_csv(STAB_RESULTS_DIR / "cifar10_representation_progression.csv")
    summary = df_prog.groupby("epoch")["ridge_test_acc"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        summary["epoch"], summary["mean"], yerr=summary["std"],
        fmt="o-", color="#3470a3", ecolor="#3470a3", elinewidth=1.5,
        capsize=4, linewidth=2, markersize=6,
        label="Ridge LS Test Accuracy (Mean ± Std, N=5)"
    )
    # Highlight Epoch 0
    ep0_val = summary.loc[summary["epoch"] == 0, "mean"].item()
    ax.axhline(ep0_val, color="#7570b3", linestyle=":", linewidth=1.5, label=f"Random Init Baseline (Epoch 0: {ep0_val:.2f}%)")

    ax.set_xlabel("Training Epoch", fontsize=11)
    ax.set_ylabel("Ridge LS Test Accuracy (%)", fontsize=11)
    ax.set_title("CIFAR-10 Representation Linear Separability Across DFA Epochs", fontsize=12, fontweight="bold")
    ax.set_ylim(32, 42)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="lower right")

    for _, row in summary.iterrows():
        ep = int(row["epoch"])
        m = row["mean"]
        ax.annotate(f"{m:.2f}%", (ep, m), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    out_path = FIG_DIR / "fig4_representation_progression.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_head_comparison():
    df_orig = pd.read_csv("data/results/cifar10_controlled_results.csv")
    df_stab = pd.read_csv(STAB_RESULTS_DIR / "cifar10_stabilization_results.csv")

    methods = ["Real Least Squares", "QUBO Binary", "STE Binary", "Binarized Least Squares", "Direct Binary"]
    orig_means = []
    orig_stds = []
    stab_means = []
    stab_stds = []

    for m in methods:
        orig_vals = df_orig[df_orig["method"] == m]["test_accuracy"]
        orig_means.append(orig_vals.mean())
        orig_stds.append(orig_vals.std())

        stab_m = "Direct Binary (Stabilized)" if m == "Direct Binary" else m
        stab_vals = df_stab[df_stab["method"] == stab_m]["test_acc"]
        stab_means.append(stab_vals.mean())
        stab_stds.append(stab_vals.std())

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, orig_means, width, yerr=orig_stds, capsize=4, label="Original Baseline (Epoch 20)", color="#d95f02", edgecolor="black")
    bars2 = ax.bar(x + width/2, stab_means, width, yerr=stab_stds, capsize=4, label="Stabilized DFA (opt_control, Epoch 20)", color="#1b9e77", edgecolor="black")

    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Classifier Head Comparison: Baseline vs. Stabilized DFA (5 Seeds)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 42)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    out_path = FIG_DIR / "fig5_head_comparison_baseline_vs_stabilized.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_loss_and_activation_comparison()
    plot_representation_progression()
    plot_head_comparison()
