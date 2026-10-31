"""Bounded validation-only grid tuning for the Bank QUBO head."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models.ls_heads import fit_ridge_classifier
from models.qubo_head import calculate_ridge_ls_scales, optimize_binary_classifier
from utils.bank.data import NUM_CLASSES
from utils.bank.metrics import compute_classification_metrics
from utils.run_provenance import collect_run_provenance, stable_run_id

DEFAULT_NUM_READS = (25, 50, 100)
DEFAULT_NUM_SWEEPS = (250, 500, 1000)
DEFAULT_SCALE_MULTIPLIERS = (0.5, 1.0, 2.0)


def _parse_grid(raw: str, cast, name: str) -> tuple:
    try:
        values = tuple(cast(part.strip()) for part in raw.split(","))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated valid values") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} must contain positive values")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} must not contain duplicates")
    return values


def validate_grid(num_reads_grid, num_sweeps_grid, scale_multipliers):
    reads = tuple(num_reads_grid)
    sweeps = tuple(num_sweeps_grid)
    multipliers = tuple(scale_multipliers)
    return tuple((int(n), int(s), float(m)) for n in reads for s in sweeps for m in multipliers)


def create_binary_targets(labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Map class indices to bipolar {-1, +1} targets."""
    return 2.0 * F.one_hot(labels.to(torch.long), num_classes=num_classes).float() - 1.0


def calculate_final_scale_and_bias(H: torch.Tensor, target: torch.Tensor, binary_weight: torch.Tensor):
    """Refit continuous scale and intercept for binary weights on training data."""
    H = H.float()
    target = target.float()
    binary_weight = binary_weight.float()

    z = H @ binary_weight
    z_mean = z.mean()
    target_mean = target.mean()

    z_centered = z - z_mean
    target_centered = target - target_mean

    denominator = z_centered @ z_centered
    if denominator.item() > 1e-12:
        scale_raw = (z_centered @ target_centered) / denominator
    else:
        scale_raw = torch.tensor(1.0, dtype=z.dtype)

    if scale_raw.item() < 0:
        binary_weight = -binary_weight
        scale = -scale_raw
    else:
        scale = scale_raw

    bias = target_mean - scale_raw * z_mean
    return binary_weight, scale, bias


def evaluate_binary_head(H: torch.Tensor, weights: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """Compute continuous scores for binary heads: H @ (W * scale) + bias."""
    return H.float() @ (weights.float() * scales.unsqueeze(0).float()) + biases.unsqueeze(0).float()


def select_configuration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select best configuration by validation balanced accuracy (or accuracy), then reads, sweeps."""
    if not rows:
        raise ValueError("Cannot select from an empty result set")
    return max(
        rows,
        key=lambda r: (
            float(r.get("validation_balanced_accuracy", r.get("validation_accuracy", 0))),
            -int(r["num_reads"]),
            -abs(float(r["scale_multiplier"]) - 1.0),
            -int(r["num_sweeps"]),
        ),
    )


def tune_from_representations(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    H_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    num_reads_grid=DEFAULT_NUM_READS,
    num_sweeps_grid=DEFAULT_NUM_SWEEPS,
    scale_multipliers=DEFAULT_SCALE_MULTIPLIERS,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]], float]:
    """Fit candidate QUBO grid on training features and select strictly on validation."""
    tuning_start = time.perf_counter()
    configurations = validate_grid(num_reads_grid, num_sweeps_grid, scale_multipliers)

    targets_train = create_binary_targets(y_train, num_classes=NUM_CLASSES)
    base_scales = calculate_ridge_ls_scales(H_train, y_train, num_classes=NUM_CLASSES, lambda_relative=1e-4)
    base_scales = torch.where(base_scales > 1e-8, base_scales, torch.full_like(base_scales, 1e-8))

    rows = []
    candidates_data = {}

    for index, (num_reads, num_sweeps, multiplier) in enumerate(configurations):
        solver_seed = (int(seed) + 1009 * (index + 1)) % (2**32)
        iter_start = time.perf_counter()
        config_key = f"r{num_reads}_s{num_sweeps}_m{multiplier:.4f}"

        scaled_alphas = base_scales * float(multiplier)

        w_bin = optimize_binary_classifier(
            H=H_train,
            targets=targets_train,
            scales=scaled_alphas,
            num_classes=NUM_CLASSES,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=solver_seed,
        )

        binary_weights_list = []
        final_scales = []
        final_biases = []

        for class_id in range(NUM_CLASSES):
            w_refit, sc, b = calculate_final_scale_and_bias(
                H=H_train,
                target=targets_train[:, class_id],
                binary_weight=w_bin[:, class_id],
            )
            binary_weights_list.append(w_refit)
            final_scales.append(sc)
            final_biases.append(b)

        w_tensor = torch.stack(binary_weights_list, dim=1)
        sc_tensor = torch.stack(final_scales)
        b_tensor = torch.stack(final_biases)

        val_scores = evaluate_binary_head(H_val, w_tensor, sc_tensor, b_tensor)
        val_preds = val_scores.argmax(dim=1).numpy()
        val_probs = torch.softmax(val_scores, dim=1)[:, 1].numpy()
        val_metrics = compute_classification_metrics(y_val.numpy(), val_preds, val_probs)

        iter_time = time.perf_counter() - iter_start

        row = {
            "num_reads": num_reads,
            "num_sweeps": num_sweeps,
            "scale_multiplier": multiplier,
            "solver_seed": solver_seed,
            "validation_accuracy": val_metrics["accuracy"],
            "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
            "validation_f1": val_metrics["f1"],
            "validation_roc_auc": val_metrics.get("roc_auc", 0.0),
            "validation_pr_auc": val_metrics.get("pr_auc", 0.0),
            "runtime_sec": iter_time,
            "config_key": config_key,
        }
        rows.append(row)
        candidates_data[config_key] = (w_tensor, sc_tensor, b_tensor)

    winning_row = select_configuration(rows)
    for r in rows:
        r["selected"] = (r["config_key"] == winning_row["config_key"])

    best_w, best_sc, best_b = candidates_data[winning_row["config_key"]]
    total_tuning_time = time.perf_counter() - tuning_start

    return winning_row, best_w, best_sc, best_b, rows, total_tuning_time
