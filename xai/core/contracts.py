"""Contracts shared by all XAI explainers and model adapters."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass
class RepresentationBundle:
    dataset: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    representation_path: str
    representation_sha256: str
    representation_source: str
    metadata: dict

@dataclass
class FittedHead:
    dataset: str
    method: str
    predict_scores: Callable[[np.ndarray], np.ndarray]
    predict_proba: Callable[[np.ndarray], np.ndarray]
    class_names: list[str]
    feature_transform: str
    fit_metadata: dict


def validate_matrix(value, name: str, *, n_features: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if n_features is not None and arr.shape[1] != n_features:
        raise ValueError(f"{name} must have {n_features} features; got {arr.shape[1]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def validate_predictor(predict_fn: Callable[[np.ndarray], np.ndarray], n_features: int) -> None:
    probe = np.zeros((2, n_features), dtype=np.float64)
    out = np.asarray(predict_fn(probe), dtype=np.float64)
    if out.ndim != 2 or out.shape[0] != 2 or out.shape[1] < 2:
        raise ValueError("predict_proba must return [n_samples, n_classes] with at least two classes")
    if not np.isfinite(out).all():
        raise ValueError("predict_proba returned non-finite values")
    if np.any(out < -1e-7) or np.any(out > 1 + 1e-7):
        raise ValueError("predict_proba values must lie in [0, 1]")
    if not np.allclose(out.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("predict_proba rows must sum to one")
