"""Validation checks for explanation arrays and local surrogate outputs."""
from __future__ import annotations
import numpy as np


def validate_explanation(values, n_features: int, name: str) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n_features,):
        raise ValueError(f"{name} shape {arr.shape} != ({n_features},)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")


def local_accuracy_status(error: float, tolerance: float = 0.03) -> str:
    if not np.isfinite(error):
        return "invalid"
    return "pass" if error <= tolerance else "review"
