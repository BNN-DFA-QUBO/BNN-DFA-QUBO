"""Shared result logging utilities for Stroke experiments.

Directly mirrors MNIST and Bank results logging so every run automatically
appends its results to data/results/stroke_results.csv.
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
except ImportError:  # pragma: no cover
    torch = None


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_CSV = Path(os.environ.get("STROKE_RESULTS_CSV", RESULTS_DIR / "stroke_results.csv"))


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
    if hasattr(model, "parameters"):
        return sum(p.numel() for p in model.parameters())
    if hasattr(model, "numel"):
        return int(model.numel())
    if hasattr(model, "size"):
        return int(model.size)
    return 0


class Timer:
    """Context manager for timing execution intervals."""
    def __init__(self) -> None:
        self.seconds: float = 0.0
        self._start: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start is not None:
            self.seconds = time.perf_counter() - self._start


def append_result(
    *,
    experiment: str,
    method: str,
    seed: int | None,
    balanced_accuracy: float,
    f1_score: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
    roc_auc: float | None = None,
    pr_auc: float | None = None,
    test_accuracy: float | None = None,
    validation_balanced_accuracy: float | None = None,
    threshold: float | None = None,
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
    """Append one Stroke result to data/results/stroke_results.csv."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    provenance = provenance or collect_run_provenance(get_device_name())
    configuration = {
        "experiment": experiment,
        "method": method,
        "seed": seed,
        "qubo_tuning_enabled": qubo_tuning_enabled,
        "qubo_num_reads_selected": qubo_num_reads_selected,
        "qubo_num_sweeps_selected": qubo_num_sweeps_selected,
        "qubo_scale_multiplier_selected": qubo_scale_multiplier_selected,
        "representation_artifact_id": representation_artifact_id,
    }
    run_id = run_id or stable_run_id(experiment, configuration, provenance)

    row = {
        "run_id": run_id,
        "experiment": experiment,
        "method": method,
        "seed": seed if seed is not None else "",
        "platform": get_device_name(),
        "hardware": get_hardware_label(),
        "balanced_accuracy": balanced_accuracy,
        "f1_score": "" if f1_score is None else f1_score,
        "precision": "" if precision is None else precision,
        "recall": "" if recall is None else recall,
        "roc_auc": "" if roc_auc is None else roc_auc,
        "pr_auc": "" if pr_auc is None else pr_auc,
        "test_accuracy": "" if test_accuracy is None else test_accuracy,
        "validation_balanced_accuracy": "" if validation_balanced_accuracy is None else validation_balanced_accuracy,
        "threshold": "" if threshold is None else threshold,
        "test_loss": "" if test_loss is None else test_loss,
        "training_time_sec": "" if training_time_sec is None else training_time_sec,
        "parameter_count": "" if parameter_count is None else parameter_count,
        "qubo_time_sec": "" if qubo_time_sec is None else qubo_time_sec,
        "qubo_selected_candidate_time_sec": "" if qubo_selected_candidate_time_sec is None else qubo_selected_candidate_time_sec,
        "backbone_training_time_sec": "" if backbone_training_time_sec is None else backbone_training_time_sec,
        "feature_extraction_time_sec": "" if feature_extraction_time_sec is None else feature_extraction_time_sec,
        "head_fit_time_sec": "" if head_fit_time_sec is None else head_fit_time_sec,
        "end_to_end_training_time_sec": "" if end_to_end_training_time_sec is None else end_to_end_training_time_sec,
        "current_run_total_time_sec": "" if current_run_total_time_sec is None else current_run_total_time_sec,
        "pipeline_timing_status": pipeline_timing_status,
        "backbone_parameter_count": "" if backbone_parameter_count is None else backbone_parameter_count,
        "head_parameter_count": "" if head_parameter_count is None else head_parameter_count,
        "total_parameter_count": "" if total_parameter_count is None else total_parameter_count,
        "notes": notes,
        "qubo_tuning_enabled": qubo_tuning_enabled,
        "qubo_num_reads_selected": "" if qubo_num_reads_selected is None else qubo_num_reads_selected,
        "qubo_num_sweeps_selected": "" if qubo_num_sweeps_selected is None else qubo_num_sweeps_selected,
        "qubo_grid_protocol": qubo_grid_protocol,
        "qubo_scale_multiplier_selected": "" if qubo_scale_multiplier_selected is None else qubo_scale_multiplier_selected,
        "qubo_validation_accuracy": "" if qubo_validation_accuracy is None else qubo_validation_accuracy,
        "qubo_validation_balanced_accuracy": "" if qubo_validation_balanced_accuracy is None else qubo_validation_balanced_accuracy,
        "qubo_grid_size": "" if qubo_grid_size is None else qubo_grid_size,
        "qubo_solver_seed_selected": "" if qubo_solver_seed_selected is None else qubo_solver_seed_selected,
        "qubo_selection_metric": qubo_selection_metric,
        "representation_artifact_id": representation_artifact_id,
        "representation_path": representation_path,
        "representation_cache_status": representation_cache_status,
        "n_train": "" if n_train is None else n_train,
        "n_validation": "" if n_validation is None else n_validation,
        "n_test": "" if n_test is None else n_test,
        "dataset_fingerprint": dataset_fingerprint,
        "split_protocol": split_protocol,
        "representation_sha256": representation_sha256,
        "backbone_training_json": backbone_training_json,
        "git_commit_sha": provenance.get("git_commit_sha", "unavailable"),
        "git_worktree_dirty": provenance.get("git_worktree_dirty", "unavailable"),
        "git_worktree_fingerprint": provenance.get("git_worktree_fingerprint", "unavailable"),
        "python_version": provenance.get("python_version", "unavailable"),
        "os_platform": provenance.get("os_platform", "unavailable"),
        "device": provenance.get("device", get_device_name()),
        "package_versions_json": json.dumps(provenance.get("package_versions", {}), sort_keys=True),
        "provenance_json": json.dumps(provenance, sort_keys=True),
    }

    fieldnames = list(row.keys())
    exists = RESULTS_CSV.exists()

    if exists:
        with RESULTS_CSV.open("r", newline="", encoding="utf-8") as current:
            reader = csv.DictReader(current)
            old_fields = list(reader.fieldnames or [])
            historical_rows = list(reader)
        # Avoid duplicate row appends if exact same run_id exists
        if "run_id" in old_fields and any(existing.get("run_id") == run_id for existing in historical_rows):
            print(f"[RESULTS] Result with run_id={run_id} already logged; skipping duplicate.")
            return
        fieldnames = old_fields + [name for name in fieldnames if name not in old_fields]
        if fieldnames != old_fields:
            temp_name = None
            try:
                with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                                                 dir=RESULTS_CSV.parent, delete=False) as temp:
                    temp_name = temp.name
                    writer = csv.DictWriter(temp, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(historical_rows)
                os.replace(temp_name, RESULTS_CSV)
            finally:
                if temp_name and os.path.exists(temp_name):
                    os.unlink(temp_name)

    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[RESULTS] Saved result to {RESULTS_CSV}")

