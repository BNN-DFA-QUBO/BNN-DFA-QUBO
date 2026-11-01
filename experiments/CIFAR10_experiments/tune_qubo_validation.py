"""Bounded validation-only grid tuning for the CIFAR-10 QUBO head.

Evaluates candidate configurations strictly on training and validation splits.
The test split is never accessed during candidate generation or selection.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from experiments.MNIST_experiments.bnn_dfa_qubo import (
    calculate_final_scale_and_bias,
    create_binary_targets,
)
from models.qubo_head import (
    calculate_ridge_ls_scales,
    optimize_binary_classifier,
)
from utils.CIFAR10.representation import DEFAULT_REPRESENTATION_DIR
from utils.run_provenance import collect_run_provenance, stable_run_id

DEFAULT_TUNING_OUTPUT_DIR = Path("data/results_diagnostic/cifar10_qubo")
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

    return tuple((n, s, float(m)) for n in reads for s in sweeps for m in multipliers)


def select_configuration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select best configuration by validation accuracy with deterministic tie-breaks."""
    if not rows:
        raise ValueError("cannot select from an empty result set")
    return min(
        rows,
        key=lambda row: (
            -float(row["validation_accuracy"]),
            int(row["num_reads"]),
            abs(float(row["scale_multiplier"]) - 1.0),
            int(row["num_sweeps"]),
            float(row["scale_multiplier"]),
        ),
    )


def apply_scale_multiplier(base_scales: torch.Tensor, multiplier: float) -> torch.Tensor:
    return base_scales * float(multiplier)


def _accuracy(scores: torch.Tensor, labels: torch.Tensor) -> float:
    return float((scores.argmax(dim=1) == labels).float().mean().item() * 100.0)


def tune_from_representations(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    H_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    num_reads_grid: tuple = DEFAULT_NUM_READS,
    num_sweeps_grid: tuple = DEFAULT_NUM_SWEEPS,
    scale_multipliers: tuple = DEFAULT_SCALE_MULTIPLIERS,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Fit candidate grid on training features and evaluate on validation features."""
    tuning_started = time.perf_counter()
    configurations = validate_grid(num_reads_grid, num_sweeps_grid, scale_multipliers)
    targets_train = create_binary_targets(y_train, num_classes=10)

    base_scales = calculate_ridge_ls_scales(
        H_train, y_train, num_classes=10, lambda_relative=1e-4
    )
    base_scales = torch.where(
        base_scales > 1e-8, base_scales, torch.full_like(base_scales, 1e-8)
    )

    rows = []
    for index, (num_reads, num_sweeps, multiplier) in enumerate(configurations):
        solver_seed = (int(seed) + 1009 * (index + 1)) % (2 ** 32)
        started = time.perf_counter()
        row: dict[str, Any] = {
            "num_reads": num_reads,
            "scale_multiplier": multiplier,
            "num_sweeps": num_sweeps,
            "solver_seed": solver_seed,
            "validation_accuracy": 0.0,
            "selected": False,
            "status": "failed",
            "error": "",
            "runtime_sec": 0.0,
        }
        try:
            scaled_alphas = apply_scale_multiplier(base_scales, multiplier)
            weights = optimize_binary_classifier(
                H=H_train,
                targets=targets_train,
                scales=scaled_alphas,
                num_classes=10,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                seed=solver_seed,
            )
            weights, final_scales, final_bias = calculate_final_scale_and_bias(
                H=H_train,
                labels=y_train,
                binary_weights=weights,
                num_classes=10,
            )
            val_scores = (
                H_val.float() @ (weights * final_scales.unsqueeze(0)) + final_bias
            )
            row.update(
                {
                    "validation_accuracy": _accuracy(val_scores, y_val),
                    "status": "ok",
                    "_weights": weights,
                    "_scales": final_scales,
                    "_bias": final_bias,
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"

        row["runtime_sec"] = time.perf_counter() - started
        rows.append(row)

        status_text = (
            f"{row['validation_accuracy']:.2f}%"
            if row["status"] == "ok"
            else f"FAILED ({row['error']})"
        )
        print(
            f"[QUBO-TUNE] reads={num_reads}, sweeps={num_sweeps}, "
            f"mult={multiplier:g} -> val_acc={status_text} ({row['runtime_sec']:.2f}s)"
        )

    successful = [r for r in rows if r["status"] == "ok"]
    winner = select_configuration(successful) if successful else None
    tuning_total_seconds = time.perf_counter() - tuning_started

    if winner is not None:
        winner["selected"] = True
        winner["selected_candidate_runtime_sec"] = winner["runtime_sec"]
        winner["tuning_total_seconds"] = tuning_total_seconds

    for r in rows:
        r["tuning_total_seconds"] = tuning_total_seconds
        r["selected_candidate_runtime_sec"] = (
            winner["runtime_sec"] if (winner is not None and r is winner) else ""
        )

    return winner, rows


def write_tuning_csv(
    rows: list[dict[str, Any]],
    seed: int,
    epochs: int,
    validation_fraction: float,
    output_dir: Path | str = DEFAULT_TUNING_OUTPUT_DIR,
    representation_artifact_id: str = "",
    n_train: int | None = None,
    n_validation: int | None = None,
    n_test: int | None = None,
) -> Path:
    """Record candidate-level QUBO validation tuning diagnostics.

    Saves configuration metadata, search grid parameters, candidate validation
    accuracies, selection status, runtimes, and errors. Test accuracy is strictly
    excluded to prevent test-set leakage in configuration selection.
    """
    if str(output_dir) == str(DEFAULT_TUNING_OUTPUT_DIR) and "CIFAR10_QUBO_TUNING_DIR" in os.environ:
        out_dir = Path(os.environ["CIFAR10_QUBO_TUNING_DIR"])
    else:
        out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    val_str = f"{float(validation_fraction):.4f}".replace(".", "p")
    art_str = str(representation_artifact_id) if representation_artifact_id else "unknown"
    csv_path = out_dir / f"cifar10_qubo_tuning_seed_{seed}_epochs_{epochs}_val_{val_str}_art_{art_str}_{stamp}.csv"

    fieldnames = [
        "seed",
        "epochs",
        "validation_fraction",
        "representation_artifact_id",
        "n_train",
        "n_validation",
        "n_test",
        "num_reads",
        "num_sweeps",
        "scale_multiplier",
        "solver_seed",
        "validation_accuracy",
        "selected",
        "runtime_sec",
        "selected_candidate_runtime_sec",
        "tuning_total_seconds",
        "status",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row_dict = {
                "seed": seed,
                "epochs": epochs,
                "validation_fraction": validation_fraction,
                "representation_artifact_id": representation_artifact_id,
                "n_train": "" if n_train is None else n_train,
                "n_validation": "" if n_validation is None else n_validation,
                "n_test": "" if n_test is None else n_test,
                "num_reads": r.get("num_reads", ""),
                "num_sweeps": r.get("num_sweeps", ""),
                "scale_multiplier": r.get("scale_multiplier", ""),
                "solver_seed": r.get("solver_seed", ""),
                "validation_accuracy": r.get("validation_accuracy", 0.0),
                "selected": r.get("selected", False),
                "runtime_sec": r.get("runtime_sec", 0.0),
                "selected_candidate_runtime_sec": r.get("selected_candidate_runtime_sec", ""),
                "tuning_total_seconds": r.get("tuning_total_seconds", ""),
                "status": r.get("status", "unknown"),
                "error": r.get("error", ""),
            }
            writer.writerow(row_dict)

    return csv_path
