"""Shared result logging utilities for MNIST experiments.

This module is intentionally independent of the existing training logic.
Use it to record final metrics without changing model implementations.
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
# Optional redirect used by diagnostics/tests to keep historical CSVs untouched.
RESULTS_CSV = Path(os.environ.get("BNN_DFA_RESULTS_CSV", RESULTS_DIR / "mnist_results.csv"))


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


def append_result(
    *,
    experiment: str,
    method: str,
    seed: int | None,
    test_accuracy: float,
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
    run_id: str | None = None,
    provenance: dict | None = None,
) -> None:
    """Append one final MNIST result to the shared CSV."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    provenance = provenance or collect_run_provenance(get_device_name())
    configuration = {
        "experiment": experiment, "method": method, "seed": seed,
        "qubo_tuning_enabled": qubo_tuning_enabled,
        "qubo_num_reads_selected": qubo_num_reads_selected,
        "qubo_num_sweeps_selected": qubo_num_sweeps_selected,
        "qubo_scale_multiplier_selected": qubo_scale_multiplier_selected,
        "qubo_validation_fraction": qubo_validation_fraction,
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
        "test_accuracy": test_accuracy,
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
        "qubo_tuning_enabled": qubo_tuning_enabled,
        "qubo_num_reads_selected": "" if qubo_num_reads_selected is None else qubo_num_reads_selected,
        "qubo_num_sweeps_selected": "" if qubo_num_sweeps_selected is None else qubo_num_sweeps_selected,
        "qubo_grid_protocol": qubo_grid_protocol,
        "qubo_scale_multiplier_selected": "" if qubo_scale_multiplier_selected is None else qubo_scale_multiplier_selected,
        "qubo_validation_accuracy": "" if qubo_validation_accuracy is None else qubo_validation_accuracy,
        "qubo_validation_fraction": "" if qubo_validation_fraction is None else qubo_validation_fraction,
        "qubo_grid_size": "" if qubo_grid_size is None else qubo_grid_size,
        "qubo_solver_seed_selected": "" if qubo_solver_seed_selected is None else qubo_solver_seed_selected,
        "qubo_selection_metric": qubo_selection_metric,
        "representation_artifact_id": representation_artifact_id,
        "representation_path": representation_path,
        "representation_cache_status": representation_cache_status,
        "representation_sha256": representation_sha256,
        "dataset_fingerprint": dataset_fingerprint,
        "split_protocol": split_protocol,
        "backbone_training_json": backbone_training_json,
        "n_train": "" if n_train is None else n_train,
        "n_validation": "" if n_validation is None else n_validation,
        "n_test": "" if n_test is None else n_test,
        "git_commit_sha": provenance.get("git_commit_sha", "unavailable"),
        "git_worktree_dirty": provenance.get("git_worktree_dirty", "unavailable"),
        "git_worktree_fingerprint": provenance.get("git_worktree_fingerprint", "unavailable"),
        "python_version": provenance.get("python_version", "unavailable"),
        "os_platform": provenance.get("os_platform", "unavailable"),
        "device": provenance.get("device", get_device_name()),
        "package_versions_json": json.dumps(provenance.get("package_versions", {}), sort_keys=True),
        "provenance_json": json.dumps(provenance, sort_keys=True),
        "notes": notes,
    }

    fieldnames = list(row.keys())
    exists = RESULTS_CSV.exists()

    # If an older shared CSV exists, retain its schema/order and append the new
    # optional columns by rewriting every historical row with blank new values.
    # This preserves old records and makes new rows readable under one header.
    if exists:
        with RESULTS_CSV.open("r", newline="", encoding="utf-8") as current:
            reader = csv.DictReader(current)
            old_fields = list(reader.fieldnames or [])
            historical_rows = list(reader)
        if "run_id" in old_fields and any(existing.get("run_id") == run_id for existing in historical_rows):
            raise FileExistsError(f"Refusing duplicate MNIST result append for run_id={run_id}")
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


class Timer:
    """Small context manager for timing training or QUBO optimization."""

    def __init__(self) -> None:
        self.seconds: float = 0.0
        self._start: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start is not None:
            self.seconds = time.perf_counter() - self._start
