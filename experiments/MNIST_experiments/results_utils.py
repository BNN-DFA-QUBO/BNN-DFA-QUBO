"""Shared result logging utilities for MNIST experiments.

This module is intentionally independent of the existing training logic.
Use it to record final metrics without changing model implementations.
"""

from __future__ import annotations

import csv
import platform
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_CSV = RESULTS_DIR / "mnist_results.csv"


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
    notes: str = "",
) -> None:
    """Append one final MNIST result to the shared CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    row = {
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
        "notes": notes,
    }

    fieldnames = list(row.keys())
    exists = RESULTS_CSV.exists()

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
