"""Numerically robust least-squares utilities for BNN-DFA-QUBO.

The ridge solver uses CPU float64 and an augmented least-squares system, not
normal equations. The intercept is not penalized. Lambda selection must use
training/validation representations only; do not choose hyperparameters on test.
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
    H = H.detach().to(device="cpu", dtype=torch.float64)
    return torch.cat(
        [H, torch.ones((H.shape[0], 1), dtype=torch.float64)],
        dim=1,
    )


def _targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.detach().to(device="cpu", dtype=torch.long)
    return F.one_hot(labels, num_classes=num_classes).to(torch.float64)


def _mean_feature_energy(H: torch.Tensor) -> torch.Tensor:
    H = H.detach().to(device="cpu", dtype=torch.float64)
    energy = (H * H).sum(dim=0).mean()
    if not torch.isfinite(energy) or energy <= 0:
        raise ValueError("Feature energy is zero or non-finite")
    return energy


def fit_ridge_regression(
    H: torch.Tensor,
    targets: torch.Tensor,
    lambda_relative: float = 1e-4,
) -> torch.Tensor:
    """Fit numerically stable ridge least-squares to arbitrary continuous targets.

    This is the shared scalar-/multi-output solver. targets may have shape
    [n_samples] or [n_samples, n_outputs]. The final intercept is unpenalized.

    The penalty is scaled as lambda_relative * mean_j(sum_i H[i,j]**2) and
    implemented by adding regularization rows to the least-squares system,
    not by forming normal equations. The solve runs on CPU float64 with the
    SVD-based gelsd driver. Returns coefficients of shape
    [n_features + 1, n_outputs].
    """
    if not torch.isfinite(torch.tensor(float(lambda_relative))) or lambda_relative < 0:
        raise ValueError("lambda_relative must be finite and >= 0")

    H64 = H.detach().to(device="cpu", dtype=torch.float64)
    Y = torch.as_tensor(targets).detach().to(device="cpu", dtype=torch.float64)
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)
    if Y.ndim != 2 or Y.shape[0] != H64.shape[0]:
        raise ValueError("targets must have shape [n_samples] or [n_samples, n_outputs]")
    if H64.ndim != 2 or H64.shape[0] == 0 or H64.shape[1] == 0:
        raise ValueError("H must be a non-empty [n_samples, n_features] matrix")
    if not torch.isfinite(H64).all() or not torch.isfinite(Y).all():
        raise ValueError("H and targets must contain only finite values")

    X = _augmented_features(H64)
    if lambda_relative == 0:
        return torch.linalg.lstsq(X, Y, driver="gelsd", rcond=1e-6).solution

    p = H64.shape[1]
    lam_actual = float(lambda_relative) * float(_mean_feature_energy(H64))
    reg_rows = torch.zeros((p, p + 1), dtype=torch.float64)
    reg_rows[torch.arange(p), torch.arange(p)] = lam_actual ** 0.5
    augmented_A = torch.cat([X, reg_rows], dim=0)
    augmented_Y = torch.cat(
        [Y, torch.zeros((p, Y.shape[1]), dtype=torch.float64)], dim=0
    )
    return torch.linalg.lstsq(
        augmented_A, augmented_Y, driver="gelsd", rcond=1e-12
    ).solution


def fit_ridge_classifier(
    H: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
    lambda_relative: float = 1e-4,
) -> torch.Tensor:
    """Fit multi-class Ridge LS with one-hot targets and an unpenalized intercept.

    Returns CPU float64 weights with shape [n_features + 1, num_classes].
    The numerical solve is delegated to fit_ridge_regression so scalar Stroke
    and multi-class MNIST heads share the same augmented SVD implementation.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    Y = _targets(labels, num_classes)
    if Y.shape[0] != H.shape[0]:
        raise ValueError("labels length must match H rows")
    return fit_ridge_regression(H, Y, lambda_relative=lambda_relative)


def predict_scores(H: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    X = _augmented_features(H)
    W = weights.detach().to(device="cpu", dtype=torch.float64)
    return X @ W


def accuracy_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> float:
    y = labels.detach().to(device="cpu", dtype=torch.long)
    return 100.0 * (scores.argmax(dim=1) == y).double().mean().item()


def select_ridge_lambda(
    H_fit: torch.Tensor,
    y_fit: torch.Tensor,
    H_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    num_classes: int = 10,
    lambdas: Iterable[float] = (1e-8, 1e-6, 1e-4),
) -> RidgeSelection:
    """Select lambda solely by validation accuracy; ties favor stronger ridge."""
    values = sorted({float(v) for v in lambdas})
    if not values or any(v <= 0 for v in values):
        raise ValueError("Provide one or more positive relative lambda values")

    scores: dict[float, float] = {}
    for lam in values:
        W = fit_ridge_classifier(H_fit, y_fit, num_classes, lam)
        scores[lam] = accuracy_from_scores(predict_scores(H_val, W), y_val)

    selected = sorted(values, key=lambda lam: (-scores[lam], -lam))[0]
    return RidgeSelection(selected, scores[selected], scores)


def fit_selected_ridge_on_all_training(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    selection: RidgeSelection,
    num_classes: int = 10,
) -> torch.Tensor:
    """Refit selected lambda on all training representations."""
    return fit_ridge_classifier(
        H_train, y_train, num_classes, selection.selected_lambda
    )
