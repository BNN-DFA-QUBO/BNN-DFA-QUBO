"""Record the output of an existing reproducibility run.

This avoids changing the existing reproducibility runner until you are ready.

Example:
    python reproducibility/record_run.py \
        --experiment bnn_dfa_qubo \
        --accuracy 97.42 \
        --seed 42 \
        --platform cuda \
        --hardware "NVIDIA RTX 4060"

For multiple machines, run this after each verified protocol run.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "results" / "reproducibility_results.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--accuracy", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--hardware", default="")
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "experiment": args.experiment,
        "seed": args.seed,
        "platform": args.platform,
        "hardware": args.hardware,
        "test_accuracy": args.accuracy,
    }

    fields = list(row.keys())
    exists = OUT.exists()

    with OUT.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
