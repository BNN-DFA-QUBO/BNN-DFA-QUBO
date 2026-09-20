"""Generate paper-ready plots for the MNIST Results and Analysis section.

Run:
    python experiments/MNIST_experiments/plot_results.py

The script uses data/results/mnist_results.csv.
It does not train any model.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "results" / "mnist_results.csv"
OUT = ROOT / "data" / "results" / "figures"


def load():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} does not exist. Run the experiments first."
        )
    df = pd.read_csv(INPUT)
    for col in [
        "test_accuracy",
        "test_loss",
        "training_time_sec",
        "parameter_count",
        "qubo_time_sec",
        "seed",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main_results(df):
    summary = (
        df.groupby("method")["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        summary["method"],
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=5,
    )
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("MNIST classification results")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save(fig, "fig1_main_classification_accuracy.png")


def head_comparison(df):
    keywords = ["least", "binary", "direct", "ste", "qubo"]
    mask = df["method"].str.lower().apply(
        lambda x: any(k in x for k in keywords)
    )
    d = df[mask].copy()

    if d.empty:
        print("No classifier-head results found; skipping head comparison.")
        return

    summary = d.groupby("method")["test_accuracy"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        summary["method"],
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=5,
    )
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Classifier-head comparison")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save(fig, "fig2_classifier_head_comparison.png")


def qubo_comparison(df):
    d = df[
        df["method"].str.contains("binary|qubo", case=False, regex=True, na=False)
    ].copy()

    if d.empty:
        print("No binary/QUBO results found; skipping QUBO comparison.")
        return

    summary = d.groupby("method")["test_accuracy"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        summary["method"],
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=5,
    )
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Binary head versus QUBO-optimized head")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save(fig, "fig3_qubo_comparison.png")


def ablation(df):
    names = [
        "BNN",
        "BNN + DFA",
        "BNN + DFA + Binary Head",
        "BNN + DFA + QUBO",
    ]
    available = df[df["method"].isin(names)].copy()

    if available.empty:
        print("No ablation rows found; skipping ablation plot.")
        return

    summary = (
        available.groupby("method")["test_accuracy"]
        .agg(["mean", "std"])
        .reindex(names)
        .dropna(how="all")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        summary["method"],
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=5,
    )
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Ablation study")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save(fig, "fig4_ablation_study.png")


def computational_cost(df):
    d = df.dropna(subset=["training_time_sec"]).copy()

    if d.empty:
        print("No training-time data found; skipping computational-cost plot.")
        return

    summary = (
        d.groupby("method")
        .agg(
            accuracy=("test_accuracy", "mean"),
            training_time=("training_time_sec", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(summary["training_time"], summary["accuracy"])
    for _, row in summary.iterrows():
        ax.annotate(
            row["method"],
            (row["training_time"], row["accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy versus training time")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    save(fig, "fig5_accuracy_vs_training_time.png")


def main():
    df = load()

    main_results(df)
    head_comparison(df)
    qubo_comparison(df)
    ablation(df)
    computational_cost(df)

    print("\nAll available MNIST result plots generated.")


if __name__ == "__main__":
    main()
