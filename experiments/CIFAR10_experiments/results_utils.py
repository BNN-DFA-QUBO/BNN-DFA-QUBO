"""Shared result logging utilities for CIFAR-10 experiments.

Records final metrics to data/results/cifar10_results.csv without modifying
any MNIST or Stroke result files.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

from utils.run_provenance import collect_run_provenance, stable_run_id

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_CSV = Path(
    os.environ.get("CIFAR10_RESULTS_CSV", RESULTS_DIR / "cifar10_results.csv")
)


def get_device_name() -> str:
    if torch is None:
        return "unknown"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_hardware_label() -> str:
    if torch is None:
        return platform.platform()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "Apple Silicon / MPS"
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA"
    return platform.processor() or platform.machine() or "CPU"


def count_parameters(model: Any) -> int:
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters())


class Timer:
    """Context manager for wall-clock execution timing."""

    def __enter__(self):
        self.start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.seconds = time.perf_counter() - self.start


FIELDNAMES = [
    "run_id",
    "timestamp",
    "experiment",
    "dataset",
    "method",
    "seed",
    "test_accuracy",
    "validation_accuracy",
    "training_accuracy",
    "test_loss",
    "training_time_sec",
    "backbone_training_time_sec",
    "feature_extraction_time_sec",
    "head_fit_time_sec",
    "end_to_end_training_time_sec",
    "current_run_total_time_sec",
    "parameter_count",
    "backbone_parameter_count",
    "head_parameter_count",
    "total_parameter_count",
    "pipeline_timing_status",
    "qubo_time_sec",
    "qubo_selected_candidate_time_sec",
    "qubo_tuning_enabled",
    "qubo_num_reads_selected",
    "qubo_num_sweeps_selected",
    "qubo_grid_protocol",
    "qubo_scale_multiplier_selected",
    "qubo_validation_accuracy",
    "qubo_validation_fraction",
    "qubo_grid_size",
    "qubo_solver_seed_selected",
    "qubo_selection_metric",
    "representation_artifact_id",
    "representation_path",
    "representation_cache_status",
    "n_train",
    "n_validation",
    "n_test",
    "dataset_fingerprint",
    "split_protocol",
    "representation_sha256",
    "backbone_training_json",
    "git_commit",
    "platform",
    "device",
    "hardware",
    "notes",
]


def append_result(
    *,
    experiment: str,
    method: str,
    seed: int | None,
    test_accuracy: float,
    validation_accuracy: float | None = None,
    training_accuracy: float | None = None,
    test_loss: float | None = None,
    training_time_sec: float | None = None,
    parameter_count: int | None = None,
    qubo_time_sec: float | None = None,
    qubo_selected_candidate_time_sec: float | None = None,
    backbone_training_time_sec: float | None = None,
    feature_extraction_time_sec: float | None = None,
    head_fit_time_sec: float | None = None,
    end_to_end_training_time_sec: float | None = None,
    current_run_total_time_sec: float | None = None,
    backbone_parameter_count: int | None = None,
    head_parameter_count: int | None = None,
    total_parameter_count: int | None = None,
    pipeline_timing_status: str = "",
    notes: str = "",
    qubo_tuning_enabled: bool = False,
    qubo_num_reads_selected: int | None = None,
    qubo_num_sweeps_selected: int | None = None,
    qubo_grid_protocol: str = "",
    qubo_scale_multiplier_selected: float | None = None,
    qubo_validation_accuracy: float | None = None,
    qubo_validation_fraction: float | None = None,
    qubo_grid_size: int | None = None,
    qubo_solver_seed_selected: int | None = None,
    qubo_selection_metric: str = "",
    representation_artifact_id: str = "",
    representation_path: str = "",
    representation_cache_status: str = "",
    n_train: int | None = None,
    n_validation: int | None = None,
    n_test: int | None = None,
    dataset_fingerprint: str = "",
    split_protocol: str = "",
    representation_sha256: str = "",
    backbone_training_json: str = "",
    results_path: Path | None = None,
) -> Path:
    target_path = Path(
        results_path
        or os.environ.get("CIFAR10_RESULTS_CSV")
        or RESULTS_CSV
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    provenance = collect_run_provenance(get_device_name())

    run_id = stable_run_id(
        experiment,
        {
            "method": method,
            "seed": seed,
            "test_accuracy": test_accuracy,
            "dataset": "CIFAR-10",
            "representation_artifact_id": representation_artifact_id,
        },
        provenance,
    )

    row = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "experiment": experiment,
        "dataset": "CIFAR-10",
        "method": method,
        "seed": seed if seed is not None else "",
        "test_accuracy": f"{test_accuracy:.4f}",
        "validation_accuracy": (
            f"{validation_accuracy:.4f}" if validation_accuracy is not None else ""
        ),
        "training_accuracy": (
            f"{training_accuracy:.4f}" if training_accuracy is not None else ""
        ),
        "test_loss": f"{test_loss:.6f}" if test_loss is not None else "",
        "training_time_sec": (
            f"{training_time_sec:.4f}" if training_time_sec is not None else ""
        ),
        "backbone_training_time_sec": (
            f"{backbone_training_time_sec:.4f}"
            if backbone_training_time_sec is not None
            else ""
        ),
        "feature_extraction_time_sec": (
            f"{feature_extraction_time_sec:.4f}"
            if feature_extraction_time_sec is not None
            else ""
        ),
        "head_fit_time_sec": (
            f"{head_fit_time_sec:.4f}" if head_fit_time_sec is not None else ""
        ),
        "end_to_end_training_time_sec": (
            f"{end_to_end_training_time_sec:.4f}"
            if end_to_end_training_time_sec is not None
            else ""
        ),
        "current_run_total_time_sec": (
            f"{current_run_total_time_sec:.4f}"
            if current_run_total_time_sec is not None
            else ""
        ),
        "parameter_count": (
            str(parameter_count) if parameter_count is not None else ""
        ),
        "backbone_parameter_count": (
            str(backbone_parameter_count)
            if backbone_parameter_count is not None
            else ""
        ),
        "head_parameter_count": (
            str(head_parameter_count) if head_parameter_count is not None else ""
        ),
        "total_parameter_count": (
            str(total_parameter_count) if total_parameter_count is not None else ""
        ),
        "pipeline_timing_status": pipeline_timing_status,
        "qubo_time_sec": (
            f"{qubo_time_sec:.4f}" if qubo_time_sec is not None else ""
        ),
        "qubo_selected_candidate_time_sec": (
            f"{qubo_selected_candidate_time_sec:.4f}"
            if qubo_selected_candidate_time_sec is not None
            else ""
        ),
        "qubo_tuning_enabled": str(qubo_tuning_enabled),
        "qubo_num_reads_selected": (
            str(qubo_num_reads_selected)
            if qubo_num_reads_selected is not None
            else ""
        ),
        "qubo_num_sweeps_selected": (
            str(qubo_num_sweeps_selected)
            if qubo_num_sweeps_selected is not None
            else ""
        ),
        "qubo_grid_protocol": qubo_grid_protocol,
        "qubo_scale_multiplier_selected": (
            f"{qubo_scale_multiplier_selected:.4f}"
            if qubo_scale_multiplier_selected is not None
            else ""
        ),
        "qubo_validation_accuracy": (
            f"{qubo_validation_accuracy:.4f}"
            if qubo_validation_accuracy is not None
            else ""
        ),
        "qubo_validation_fraction": (
            f"{qubo_validation_fraction:.4f}"
            if qubo_validation_fraction is not None
            else ""
        ),
        "qubo_grid_size": str(qubo_grid_size) if qubo_grid_size is not None else "",
        "qubo_solver_seed_selected": (
            str(qubo_solver_seed_selected)
            if qubo_solver_seed_selected is not None
            else ""
        ),
        "qubo_selection_metric": qubo_selection_metric,
        "representation_artifact_id": representation_artifact_id,
        "representation_path": representation_path,
        "representation_cache_status": representation_cache_status,
        "n_train": str(n_train) if n_train is not None else "",
        "n_validation": str(n_validation) if n_validation is not None else "",
        "n_test": str(n_test) if n_test is not None else "",
        "dataset_fingerprint": dataset_fingerprint,
        "split_protocol": split_protocol,
        "representation_sha256": representation_sha256,
        "backbone_training_json": backbone_training_json,
        "git_commit": provenance.get("git_commit", ""),
        "platform": provenance.get("platform", platform.platform()),
        "device": provenance.get("device", get_device_name()),
        "hardware": get_hardware_label(),
        "notes": notes,
    }

    file_exists = target_path.exists()
    with open(target_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[RESULTS] Saved CIFAR-10 result to {target_path}")
    return target_path
