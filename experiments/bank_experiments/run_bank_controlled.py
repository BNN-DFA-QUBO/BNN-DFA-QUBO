"""Alias entry point for Bank controlled / shared-backbone experiment."""
from __future__ import annotations

from experiments.bank_experiments.run_bank_shared_validation import run_bank_shared_validation

if __name__ == "__main__":
    import argparse
    from experiments.bank_experiments.tune_qubo_validation import (
        DEFAULT_NUM_READS,
        DEFAULT_NUM_SWEEPS,
        DEFAULT_SCALE_MULTIPLIERS,
        _parse_grid,
    )

    parser = argparse.ArgumentParser(description="Protocol 2: Controlled Shared Representation Comparison for Bank Dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-reads-grid", type=lambda s: _parse_grid(s, int, "num-reads-grid"), default=DEFAULT_NUM_READS)
    parser.add_argument("--num-sweeps-grid", type=lambda s: _parse_grid(s, int, "num-sweeps-grid"), default=DEFAULT_NUM_SWEEPS)
    parser.add_argument("--scale-multipliers", type=lambda s: _parse_grid(s, float, "scale-multipliers"), default=DEFAULT_SCALE_MULTIPLIERS)
    args = parser.parse_args()

    run_bank_shared_validation(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_reads_grid=args.num_reads_grid,
        num_sweeps_grid=args.num_sweeps_grid,
        scale_multipliers=args.scale_multipliers,
    )

