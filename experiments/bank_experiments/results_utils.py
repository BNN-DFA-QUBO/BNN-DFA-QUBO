"""Shared result logging utilities for Bank experiments."""
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
RESULTS_CSV = Path(os.environ.get("BANK_RESULTS_CSV", RESULTS_DIR / "bank_results.csv"))


class Timer:
    """Context manager measuring wall-clock execution time."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.perf_counter()
        self.seconds = self._end - self._start


def get_device_name() -> str:
    if torch is None:
        return "unknown"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_hardware_label() -> str:
    if torch is None:
        return platform.platform()
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "Apple Silicon / MPS"
    return platform.processor() or platform.machine() or "CPU"


def count_parameters(model: Any) -> int:
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters())


FIELDNAMES = [
    "run_id",
    "experiment",
    "method",
    "seed",
    "platform",
    "hardware",
    "test_accuracy",
    "balanced_accuracy",
    "f1_score",
    "precision",
    "recall",
    "roc_auc",
    "pr_auc",
    "test_loss",
    "training_time_sec",
    "parameter_count",
    "qubo_time_sec",
    "qubo_selected_candidate_time_sec",
    "backbone_training_time_sec",
    "feature_extraction_time_sec",
    "head_fit_time_sec",
    "end_to_end_training_time_sec",
    "current_run_total_time_sec",
    "pipeline_timing_status",
    "backbone_parameter_count",
    "head_parameter_count",
    "total_parameter_count",
    "notes",
    "qubo_tuning_enabled",
    "qubo_num_reads_selected",
    "qubo_num_sweeps_selected",
    "qubo_grid_protocol",
    "qubo_scale_multiplier_selected",
    "qubo_validation_accuracy",
    "qubo_validation_balanced_accuracy",
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
]


def append_bank_result(
    *,
    experiment: str,
    method: str,
    seed: int | None,
    test_accuracy: float,
    balanced_accuracy: float | None = None,
    f1_score: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
    roc_auc: float | None = None,
    pr_auc: float | None = None,
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
    pipeline_timing_status: str = "complete",
    notes: str = "",
    qubo_tuning_enabled: bool = False,
    qubo_num_reads_selected: int | None = None,
    qubo_num_sweeps_selected: int | None = None,
    qubo_grid_protocol: str = "",
    qubo_scale_multiplier_selected: float | None = None,
    qubo_validation_accuracy: float | None = None,
    qubo_validation_balanced_accuracy: float | None = None,
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
    run_id: str | None = None,
    provenance: dict | None = None,
) -> None:
    """Append one bank result to the CSV logger."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    provenance = provenance or collect_run_provenance(get_device_name())

    config = {
        "experiment": experiment,
        "method": method,
        "seed": seed,
        "qubo_tuning_enabled": qubo_tuning_enabled,
        "representation_artifact_id": representation_artifact_id,
    }
    run_id = run_id or stable_run_id(experiment, config, provenance)

    row = {
        "run_id": run_id,
        "experiment": experiment,
        "method": method,
        "seed": seed if seed is not None else "",
        "platform": get_device_name(),
        "hardware": get_hardware_label(),
        "test_accuracy": f"{test_accuracy:.2f}",
        "balanced_accuracy": f"{balanced_accuracy:.2f}" if balanced_accuracy is not None else "",
        "f1_score": f"{f1_score:.2f}" if f1_score is not None else "",
        "precision": f"{precision:.2f}" if precision is not None else "",
        "recall": f"{recall:.2f}" if recall is not None else "",
        "roc_auc": f"{roc_auc:.2f}" if roc_auc is not None else "",
        "pr_auc": f"{pr_auc:.2f}" if pr_auc is not None else "",
        "test_loss": f"{test_loss:.4f}" if test_loss is not None else "",
        "training_time_sec": f"{training_time_sec:.4f}" if training_time_sec is not None else "",
        "parameter_count": parameter_count if parameter_count is not None else "",
        "qubo_time_sec": f"{qubo_time_sec:.4f}" if qubo_time_sec is not None else "",
        "qubo_selected_candidate_time_sec": f"{qubo_selected_candidate_time_sec:.4f}" if qubo_selected_candidate_time_sec is not None else "",
        "backbone_training_time_sec": f"{backbone_training_time_sec:.4f}" if backbone_training_time_sec is not None else "",
        "feature_extraction_time_sec": f"{feature_extraction_time_sec:.4f}" if feature_extraction_time_sec is not None else "",
        "head_fit_time_sec": f"{head_fit_time_sec:.4f}" if head_fit_time_sec is not None else "",
        "end_to_end_training_time_sec": f"{end_to_end_training_time_sec:.4f}" if end_to_end_training_time_sec is not None else "",
        "current_run_total_time_sec": f"{current_run_total_time_sec:.4f}" if current_run_total_time_sec is not None else "",
        "pipeline_timing_status": pipeline_timing_status,
        "backbone_parameter_count": backbone_parameter_count if backbone_parameter_count is not None else "",
        "head_parameter_count": head_parameter_count if head_parameter_count is not None else "",
        "total_parameter_count": total_parameter_count if total_parameter_count is not None else "",
        "notes": notes,
        "qubo_tuning_enabled": qubo_tuning_enabled,
        "qubo_num_reads_selected": qubo_num_reads_selected if qubo_num_reads_selected is not None else "",
        "qubo_num_sweeps_selected": qubo_num_sweeps_selected if qubo_num_sweeps_selected is not None else "",
        "qubo_grid_protocol": qubo_grid_protocol,
        "qubo_scale_multiplier_selected": qubo_scale_multiplier_selected if qubo_scale_multiplier_selected is not None else "",
        "qubo_validation_accuracy": f"{qubo_validation_accuracy:.2f}" if qubo_validation_accuracy is not None else "",
        "qubo_validation_balanced_accuracy": f"{qubo_validation_balanced_accuracy:.2f}" if qubo_validation_balanced_accuracy is not None else "",
        "qubo_grid_size": qubo_grid_size if qubo_grid_size is not None else "",
        "qubo_solver_seed_selected": qubo_solver_seed_selected if qubo_solver_seed_selected is not None else "",
        "qubo_selection_metric": qubo_selection_metric,
        "representation_artifact_id": representation_artifact_id,
        "representation_path": representation_path,
        "representation_cache_status": representation_cache_status,
        "n_train": n_train if n_train is not None else "",
        "n_validation": n_validation if n_validation is not None else "",
        "n_test": n_test if n_test is not None else "",
        "dataset_fingerprint": dataset_fingerprint,
        "split_protocol": split_protocol,
        "representation_sha256": representation_sha256,
        "backbone_training_json": backbone_training_json,
    }

    file_exists = RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

