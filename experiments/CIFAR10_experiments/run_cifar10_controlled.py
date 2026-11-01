"""Multi-seed controlled experiment runner for CIFAR-10 BNN+DFA head comparisons.

Runs all five classifier heads across seeds:
    42, 123, 456, 789, 999
using structured function returns rather than regex parsing on stdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from experiments.CIFAR10_experiments.run_cifar10_shared_validation import (
    run_cifar10_shared_validation,
)
from utils.CIFAR10.representation import DEFAULT_REPRESENTATION_DIR

DEFAULT_SEEDS = [42, 123, 456, 789, 999]
DEFAULT_SUMMARY_CSV = Path("data/results/cifar10_controlled_results.csv")


def run_controlled_experiment(
    seeds: List[int] = DEFAULT_SEEDS,
    epochs: int = 20,
    ste_epochs: int = 50,
    direct_steps: int = 200,
    validation_fraction: float = 0.1,
    representation_dir: Path | str = DEFAULT_REPRESENTATION_DIR,
    fixed_qubo: bool = False,
    output_csv: Path | str = DEFAULT_SUMMARY_CSV,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Execute multi-seed head comparison and aggregate summary statistics."""
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file {out_path} already exists. Specify --overwrite to replace it."
        )

    all_records: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"\n=======================================================")
        print(f"Executing Seed {seed} in Controlled Experiment")
        print(f"=======================================================\n")

        head_results = run_cifar10_shared_validation(
            seed=seed,
            epochs=epochs,
            ste_epochs=ste_epochs,
            direct_steps=direct_steps,
            validation_fraction=validation_fraction,
            representation_dir=representation_dir,
            fixed_qubo=fixed_qubo,
        )

        for res in head_results:
            all_records.append({
                "seed": seed,
                "method": res["head"],
                "train_accuracy": res["train"],
                "validation_accuracy": res["val"],
                "test_accuracy": res["test"],
                "head_fit_time_sec": res["fit_time"],
                "head_parameters": res["head_params"],
            })

    df = pd.DataFrame(all_records)
    df.to_csv(out_path, index=False)
    print(f"\n[CONTROLLED] Saved detailed controlled run records to {out_path}")

    # Summary table: mean, std, min, max, count
    summary = (
        df.groupby("method")["test_accuracy"]
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

    print("\n" + "=" * 80)
    print("CIFAR-10 Controlled Multi-Seed Experiment Aggregate Summary:")
    print("-" * 80)
    print(summary.to_string(index=False))
    print("=" * 80 + "\n")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run CIFAR-10 controlled experiments across multiple seeds."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to execute (default: 42 123 456 789 999)",
    )
    parser.add_argument(
        "--single-seed",
        type=int,
        default=None,
        help="Run just one seed for debugging",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ste-epochs", type=int, default=50)
    parser.add_argument("--direct-steps", type=int, default=200)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--fixed-qubo", action="store_true")
    parser.add_argument(
        "--representation-dir",
        type=Path,
        default=DEFAULT_REPRESENTATION_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seeds = [args.single_seed] if args.single_seed is not None else args.seeds

    run_controlled_experiment(
        seeds=seeds,
        epochs=args.epochs,
        ste_epochs=args.ste_epochs,
        direct_steps=args.direct_steps,
        validation_fraction=args.validation_fraction,
        representation_dir=args.representation_dir,
        fixed_qubo=args.fixed_qubo,
        output_csv=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
