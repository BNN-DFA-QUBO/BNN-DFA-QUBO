"""LS heads for MNIST: legacy float32 LS and validation-selected ridge LS.

The selector is intended for research diagnostics and uses only a fit/validation
split of the training representations. The held-out test set is not used to
select lambda. Use select_ridge_lambda() on training data, then refit the chosen
lambda on the complete training representation before final test evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class RidgeSelection:
    selected_lambda: float
    validation_accuracy: float
    scores: dict[float, float]


def _augmented_features(H: torch.Tensor) -> torch.Tensor:
    """Append intercept column; preserve caller's device and use float64."""
    H = H.detach().to(dtype=torch.float64)
    return torch.cat(
        [H, torch.ones((H.shape[0], 1), dtype=H.dtype, device=H.device)],
        dim=1,
    )


def _targets(labels: torch.Tensor, num_classes: int, *, dtype, device):
    return F.one_hot(labels.to(device=device, dtype=torch.long), num_classes).to(dtype=dtype)


def fit_ridge_classifier(
    H: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
    lambda_relative: float = 1e-4,
) -> torch.Tensor:
    """Fit multi-output ridge LS with an unpenalized intercept.

    lambda_actual = lambda_relative * mean(diag(H.T @ H)).
    Solve on CPU float64 for numerically robust, predictable behavior across
    machines; returned matrix is CPU float64 with shape [n_features+1, classes].
    """
    if lambda_relative < 0:
        raise ValueError("lambda_relative must be >= 0")

    H64 = H.detach().to(device="cpu", dtype=torch.float64)
    y = labels.detach().to(device="cpu", dtype=torch.long)
    X = _augmented_features(H64)
    Y = _targets(y, num_classes, dtype=torch.float64, device="cpu")

    # lambda=0 is least squares and is handled by an SVD driver, not normal
    # equations, to avoid squaring the condition number.
    if lambda_relative == 0:
        return torch.linalg.lstsq(X, Y, driver="gelsd", rcond=1e-6).solution

    gram = X.T @ X
    rhs = X.T @ Y
    mean_feature_energy = gram.diagonal()[:-1].mean()
    if not torch.isfinite(mean_feature_energy) or mean_feature_energy <= 0:
        raise ValueError("Feature energy is zero or non-finite")

    reg = torch.zeros_like(gram)
    idx = torch.arange(gram.shape[0] - 1)
    reg[idx, idx] = lambda_relative * mean_feature_energy
    # Avoid explicit inverse; torch.linalg.solve uses a factorization.
    return torch.linalg.solve(gram + reg, rhs)


def predict_scores(H: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return class scores on CPU float64."""
    H64 = H.detach().to(device="cpu", dtype=torch.float64)
    X = _augmented_features(H64)
    W = weights.detach().to(device="cpu", dtype=torch.float64)
    return X @ W


def accuracy_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels_cpu = labels.detach().to(device="cpu", dtype=torch.long)
    return 100.0 * (scores.argmax(dim=1) == labels_cpu).double().mean().item()


def select_ridge_lambda(
    H_fit: torch.Tensor,
    y_fit: torch.Tensor,
    H_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    num_classes: int = 10,
    lambdas: Iterable[float] = (1e-8, 1e-6, 1e-4),
) -> RidgeSelection:
    """Select lambda solely by validation accuracy.

    Ties are resolved deterministically in favor of stronger regularization.
    """
    values = sorted({float(v) for v in lambdas})
    if not values or any(v <= 0 for v in values):
        raise ValueError("Provide one or more positive relative lambda values")

    scores: dict[float, float] = {}
    for lam in values:
        W = fit_ridge_classifier(H_fit, y_fit, num_classes, lam)
        scores[lam] = accuracy_from_scores(predict_scores(H_val, W), y_val)

    selected = sorted(values, key=lambda lam: (-scores[lam], -lam))[0]
    return RidgeSelection(
        selected_lambda=selected,
        validation_accuracy=scores[selected],
        scores=scores,
    )


def fit_selected_ridge_on_all_training(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    selection: RidgeSelection,
    num_classes: int = 10,
) -> torch.Tensor:
    """Refit selected model on all train representations; do not inspect test here."""
    return fit_ridge_classifier(
        H_train, y_train, num_classes=num_classes,
        lambda_relative=selection.selected_lambda,
    )
