"""Bounded validation-only grid tuning for the Stroke QUBO head.

Evaluates candidate configurations strictly on training and validation splits.
The test split is never accessed during candidate generation or selection.

Example:
    python -m experiments.stroke_experiments.tune_qubo_validation --seed 42
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from utils.stroke_config import (
    QUBO_NUM_READS,
    QUBO_NUM_SWEEPS,
    QUBO_RANDOM_SEED_STARTS,
    QUBO_ALPHA_MULTIPLIERS,
    QUBO_TOP_CANDIDATES_PER_SCALE,
    SEEDS,
    SPLIT_SEED,
    BALANCING_METHOD,
)
from utils.stroke_data import DATA_DIR
from utils.stroke_binary_features import standardize_hidden_features
from utils.stroke_metrics import find_best_threshold, calculate_metrics
from utils.stroke_representation import load_stroke_head_representation
from utils.run_provenance import collect_run_provenance, stable_run_id
from experiments.stroke_experiments.stroke_dfa_qubo import (
    calculate_alpha,
    sample_binary_candidates,
    recover_bias,
    _validate_positive_integer,
    _validate_seed,
    _validate_positive_real,
    _MAX_SOLVER_SEED,
)

DEFAULT_TUNING_OUTPUT_DIR = Path("data/results_diagnostic/stroke_qubo")
DEFAULT_NUM_READS = (25, 50, 100)
DEFAULT_NUM_SWEEPS = (250, 500, 1000)
DEFAULT_SCALE_MULTIPLIERS = (0.5, 1.0, 2.0)


def _parse_grid(raw: str, cast: Any, name: str) -> tuple:
    try:
        values = tuple(cast(part.strip()) for part in raw.split(","))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be comma-separated valid values"
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} must contain positive values")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} must not contain duplicates")
    return values


def validate_grid(num_reads_grid, num_sweeps_grid, scale_multipliers):
    reads = tuple(num_reads_grid)
    sweeps = tuple(num_sweeps_grid)
    multipliers = tuple(scale_multipliers)

    if not reads or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in reads):
        raise ValueError("num_reads grid must contain positive integers")
    if not sweeps or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in sweeps):
        raise ValueError("num_sweeps grid must contain positive integers")
    if not multipliers or any(not np.isfinite(v) or v <= 0 for v in multipliers):
        raise ValueError("scale multipliers must contain positive finite numbers")
    if len(set(reads)) != len(reads) or len(set(multipliers)) != len(multipliers):
        raise ValueError("grid values must not contain duplicates")
    if len(set(sweeps)) != len(sweeps):
        raise ValueError("num_sweeps grid must not contain duplicates")

    return tuple((int(n), int(s), float(m)) for n in reads for s in sweeps for m in multipliers)


def select_configuration(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select best configuration by validation composite score with deterministic tie-breaks.
    
    Ranking order:
    1. Highest validation_selection_score (or validation_balanced_accuracy)
    2. Highest validation_balanced_accuracy
    3. Fewest num_reads
    4. scale_multiplier closest to 1.0
    5. Fewest num_sweeps
    6. Lowest scale_multiplier
    """
    if not rows:
        raise ValueError("cannot select from an empty result set")
    
    valid_rows = [r for r in rows if r.get("status") == "ok"]
    if not valid_rows:
        raise RuntimeError("No candidate configuration completed successfully")

    return min(
        valid_rows,
        key=lambda row: (
            -float(row.get("validation_selection_score", row.get("validation_balanced_accuracy", 0.0))),
            -float(row.get("validation_balanced_accuracy", 0.0)),
            int(row["num_reads"]),
            abs(float(row["scale_multiplier"]) - 1.0),
            int(row["num_sweeps"]),
            float(row["scale_multiplier"]),
        ),
    )


def tune_from_representations(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    H_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    num_reads_grid: Tuple[int, ...] = DEFAULT_NUM_READS,
    num_sweeps_grid: Tuple[int, ...] = DEFAULT_NUM_SWEEPS,
    scale_multipliers: Tuple[float, ...] = DEFAULT_SCALE_MULTIPLIERS,
) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    """Fit candidate configurations from training features and evaluate on validation only.
    
    The test split is never touched in this function.
    """
    tuning_started = time.perf_counter()
    configurations = validate_grid(num_reads_grid, num_sweeps_grid, scale_multipliers)

    y_train_t = torch.as_tensor(y_train).float().reshape(-1)
    y_val_t = torch.as_tensor(y_val).float().reshape(-1)
    target_train = 2.0 * y_train_t - 1.0

    base_alpha = calculate_alpha(H_train, y_train_t)

    rows: List[Dict[str, Any]] = []

    for index, (num_reads, num_sweeps, multiplier) in enumerate(configurations):
        solver_seed = (int(seed) + 104729 * (index + 1)) % (_MAX_SOLVER_SEED + 1)
        started = time.perf_counter()
        row: Dict[str, Any] = {
            "num_reads": num_reads,
            "scale_multiplier": multiplier,
            "num_sweeps": num_sweeps,
            "solver_seed": solver_seed,
            "status": "failed",
            "error": "",
            "runtime_sec": 0.0,
            "validation_balanced_accuracy": 0.0,
            "validation_selection_score": 0.0,
            "validation_f1": 0.0,
            "validation_pr_auc": 0.0,
            "validation_roc_auc": 0.0,
            "selected": False,
        }

        try:
            candidates = sample_binary_candidates(
                H=H_train,
                target=target_train,
                base_alpha=base_alpha,
                num_reads=num_reads,
                seed=solver_seed,
                alpha_multipliers=(multiplier,),
                top_k=QUBO_TOP_CANDIDATES_PER_SCALE,
                num_sweeps=num_sweeps,
                random_seed_starts=QUBO_RANDOM_SEED_STARTS,
            )

            best_cand = None
            for cand in candidates:
                weights = cand["weights"]
                alpha = cand["alpha"]
                bias = recover_bias(H_train, target_train, weights, alpha)
                val_scores = alpha * (H_val @ weights) + bias
                threshold, val_ba = find_best_threshold(y_val_t.numpy(), val_scores.numpy())
                val_metrics = calculate_metrics(y_val_t.numpy(), val_scores.numpy(), threshold)

                val_selection_score = float(np.mean([
                    val_ba,
                    val_metrics["f1"],
                    val_metrics["pr_auc"],
                    val_metrics["roc_auc"],
                ]))

                cand_key = (
                    val_selection_score,
                    float(val_ba),
                    float(val_metrics["f1"]),
                    float(val_metrics["pr_auc"]),
                    float(val_metrics["roc_auc"]),
                )

                if best_cand is None or cand_key > best_cand["selection_key"]:
                    best_cand = {
                        "selection_key": cand_key,
                        "validation_selection_score": val_selection_score,
                        "candidate": cand,
                        "weights": weights,
                        "alpha": alpha,
                        "bias": float(bias),
                        "threshold": float(threshold),
                        "validation_balanced_accuracy": float(val_ba),
                        "validation_f1": float(val_metrics["f1"]),
                        "validation_pr_auc": float(val_metrics["pr_auc"]),
                        "validation_roc_auc": float(val_metrics["roc_auc"]),
                    }

            runtime = time.perf_counter() - started
            row.update({
                "status": "ok",
                "runtime_sec": runtime,
                "validation_balanced_accuracy": best_cand["validation_balanced_accuracy"],
                "validation_selection_score": best_cand["validation_selection_score"],
                "validation_f1": best_cand["validation_f1"],
                "validation_pr_auc": best_cand["validation_pr_auc"],
                "validation_roc_auc": best_cand["validation_roc_auc"],
                "threshold": best_cand["threshold"],
                "alpha": best_cand["alpha"],
                "base_alpha": base_alpha,
                "bias": best_cand["bias"],
                "energy": float(best_cand["candidate"]["energy"]),
                "candidate_source": best_cand["candidate"]["candidate_source"],
                "candidate_count": len(candidates),
                "_weights": best_cand["weights"],
                "_alpha": best_cand["alpha"],
                "_bias": best_cand["bias"],
                "_threshold": best_cand["threshold"],
            })

        except Exception as exc:
            row["runtime_sec"] = time.perf_counter() - started
            row["error"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)

    total_tuning_seconds = time.perf_counter() - tuning_started
    for r in rows:
        r["tuning_total_seconds"] = total_tuning_seconds

    winner = None
    try:
        winner = select_configuration(rows)
        winner["selected"] = True
        winner["selected_candidate_runtime_sec"] = winner["runtime_sec"]
        for r in rows:
            r["selected_candidate_runtime_sec"] = winner["runtime_sec"]
    except Exception:
        pass

    return winner, rows


def write_tuning_csv(
    rows: List[Dict[str, Any]],
    seed: int,
    output_dir: Path | str = DEFAULT_TUNING_OUTPUT_DIR,
    artifact_id: str = "",
    n_train: int = 0,
    n_val: int = 0,
    n_test: int = 0,
) -> Path:
    """Save tuning diagnostic records to an isolated CSV without test metrics."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    art_tag = f"_art_{artifact_id}" if artifact_id else ""
    csv_path = out_dir / f"stroke_qubo_tuning_seed_{seed}{art_tag}_{timestamp}.csv"

    fieldnames = [
        "seed",
        "num_reads",
        "num_sweeps",
        "scale_multiplier",
        "solver_seed",
        "validation_selection_score",
        "validation_balanced_accuracy",
        "validation_f1",
        "validation_pr_auc",
        "validation_roc_auc",
        "threshold",
        "alpha",
        "base_alpha",
        "bias",
        "energy",
        "candidate_source",
        "candidate_count",
        "selected",
        "status",
        "error",
        "runtime_sec",
        "tuning_total_seconds",
        "selected_candidate_runtime_sec",
        "n_train",
        "n_validation",
        "n_test",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            entry = dict(r)
            entry["seed"] = seed
            entry["n_train"] = n_train
            entry["n_validation"] = n_val
            entry["n_test"] = n_test
            writer.writerow(entry)

    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", type=Path, default=DATA_DIR / "stroke_dfa_representation_seed_42.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TUNING_OUTPUT_DIR)
    parser.add_argument(
        "--num-reads-grid",
        type=lambda s: _parse_grid(s, int, "num-reads-grid"),
        default=DEFAULT_NUM_READS,
    )
    parser.add_argument(
        "--num-sweeps-grid",
        type=lambda s: _parse_grid(s, int, "num-sweeps-grid"),
        default=DEFAULT_NUM_SWEEPS,
    )
    parser.add_argument(
        "--scale-multipliers",
        type=lambda s: _parse_grid(s, float, "scale-multipliers"),
        default=DEFAULT_SCALE_MULTIPLIERS,
    )
    args = parser.parse_args()

    data = load_stroke_head_representation(args.representation)
    H_train, H_val, H_test, transform = standardize_hidden_features(
        data["H_train"], data["H_val"], data["H_test"]
    )
    y_train = torch.as_tensor(data["y_train"]).float().reshape(-1)
    y_val = torch.as_tensor(data["y_val"]).float().reshape(-1)
    y_test = torch.as_tensor(data["y_test"]).float().reshape(-1)

    winner, rows = tune_from_representations(
        H_train=H_train,
        y_train=y_train,
        H_val=H_val,
        y_val=y_val,
        seed=args.seed,
        num_reads_grid=args.num_reads_grid,
        num_sweeps_grid=args.num_sweeps_grid,
        scale_multipliers=args.scale_multipliers,
    )

    if winner is None:
        raise RuntimeError("QUBO validation tuning failed to select any configuration")

    csv_path = write_tuning_csv(
        rows=rows,
        seed=args.seed,
        output_dir=args.output_dir,
        artifact_id=data.get("representation_sha256", "")[:12],
        n_train=len(H_train),
        n_val=len(H_val),
        n_test=len(H_test),
    )

    # Evaluate winner on test set exactly once
    test_scores = winner["_alpha"] * (H_test @ winner["_weights"]) + winner["_bias"]
    test_metrics = calculate_metrics(y_test.numpy(), test_scores.numpy(), winner["_threshold"])

    print("\n" + "=" * 60)
    print("STROKE QUBO VALIDATION TUNING COMPLETE")
    print("=" * 60)
    print(f"Selected configuration: num_reads={winner['num_reads']}, num_sweeps={winner['num_sweeps']}, scale_multiplier={winner['scale_multiplier']}")
    print(f"Validation Selection Score: {winner['validation_selection_score']:.4f}")
    print(f"Validation Balanced Accuracy: {winner['validation_balanced_accuracy'] * 100:.2f}%")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"Test F1: {test_metrics['f1'] * 100:.2f}%")
    print(f"Test ROC-AUC: {test_metrics['roc_auc'] * 100:.2f}%")
    print(f"Tuning diagnostics saved to: {csv_path}")


if __name__ == "__main__":
    main()

