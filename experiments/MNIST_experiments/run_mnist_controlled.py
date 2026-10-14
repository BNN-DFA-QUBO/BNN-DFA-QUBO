import argparse
import re
import runpy
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MNIST_EXPERIMENTS_DIR = ROOT / "experiments" / "MNIST_experiments"
RESULTS_PATH = ROOT / "data" / "mnist_controlled_results.csv"

# The repository's MNIST scripts use slightly different final labels.
# These patterns identify the test accuracy that represents each experiment.
TARGET_PATTERNS = {
    "baseline_ann": r"Test Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    "bnn_bp": r"Test Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    "bnn_dfa": r"Final Test Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    "dfa_least_squares": r"(?:Ridge LS|Least-Squares) Test Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    "dfa_binary_head": r"Binary Head Test Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    "dfa_direct_binary_head": r"BNN \+ DFA \+ Direct Binary Head Test Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    "dfa_binary_head_ste": r"BNN \+ DFA \+ STE Binary Head Test Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    "bnn_dfa_qubo": r"BNN \+ DFA \+ QUBO Binary Head Test Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
}

DISPLAY_NAMES = {
    "baseline_ann": "Standard ANN",
    "bnn_bp": "BNN + Backpropagation",
    "bnn_dfa": "BNN + DFA",
    "dfa_least_squares": "DFA + Ridge LS Head",
    "dfa_binary_head": "DFA + Binarized Ridge LS Head (refit scale+bias)",
    "dfa_direct_binary_head": "DFA + Direct Binary Head",
    "dfa_binary_head_ste": "DFA + STE Binary Head",
    "bnn_dfa_qubo": "DFA + QUBO Binary Head (Ridge LS scale init)",
}

DEFAULT_EXPERIMENTS = list(TARGET_PATTERNS)


def run_one(experiment, seed):
    path = MNIST_EXPERIMENTS_DIR / f"{experiment}.py"
    if not path.exists():
        raise FileNotFoundError(path)

    old_argv = sys.argv[:]
    captured = StringIO()

    try:
        sys.argv = [str(path), "--seed", str(seed)]
        with redirect_stdout(captured):
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = old_argv

    output = captured.getvalue()
    pattern = TARGET_PATTERNS[experiment]
    matches = re.findall(pattern, output, flags=re.IGNORECASE)

    if not matches:
        raise RuntimeError(
            f"Could not extract test accuracy for {experiment}.\n"
            f"Expected pattern: {pattern}\n\n"
            f"Last output:\n{output[-4000:]}"
        )

    accuracy = float(matches[-1])

    print(output, end="")
    print(f"[MNIST-CONTROLLED] {experiment} | seed={seed} | test_accuracy={accuracy:.2f}%")

    return {
        "experiment": experiment,
        "method": DISPLAY_NAMES[experiment],
        "seed": seed,
        "test_accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run MNIST experiments across multiple seeds and save structured results."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2024, 7, 99],
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=DEFAULT_EXPERIMENTS,
        default=DEFAULT_EXPERIMENTS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help="Raw CSV output path (default: data/mnist_controlled_results.csv)",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Summary CSV path (default: <output-stem>_summary.csv beside raw CSV)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the selected raw/summary CSVs if they already exist",
    )
    args = parser.parse_args()

    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    summary_path = args.summary_output
    if summary_path is None:
        summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    elif not summary_path.is_absolute():
        summary_path = ROOT / summary_path

    existing = [str(path) for path in (output_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to replace existing controlled MNIST outputs: "
            + ", ".join(existing)
            + ". Choose new --output/--summary-output paths or pass --overwrite."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        print("\\n" + "=" * 70)
        print(f"MNIST CONTROLLED RUN — SEED {seed}")
        print("=" * 70)

        for experiment in args.experiments:
            print("\\n" + "-" * 70)
            print(f"Running {DISPLAY_NAMES[experiment]}")
            print("-" * 70)
            rows.append(run_one(experiment, seed))

    if not rows:
        raise RuntimeError("No experiment results were collected.")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    summary = (
        df.groupby(["experiment", "method"])["test_accuracy"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "runs", "mean": "accuracy_mean", "std": "accuracy_std",
                         "min": "accuracy_min", "max": "accuracy_max"})
    )
    summary.to_csv(summary_path, index=False)

    print("\\n" + "=" * 70)
    print("MNIST CONTROLLED EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Raw results: {output_path}")
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()
