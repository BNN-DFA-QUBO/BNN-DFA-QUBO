"""Evaluation metrics and threshold tuning for Bank marketing classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    y_scores: np.ndarray | list | None = None,
) -> dict[str, float]:
    """Compute comprehensive classification metrics for tabular binary classification."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)

    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    prec = float(precision_score(y_true, y_pred, zero_division=0) * 100.0)
    rec = float(recall_score(y_true, y_pred, zero_division=0) * 100.0)
    f1 = float(f1_score(y_true, y_pred, zero_division=0) * 100.0)

    res: dict[str, float] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    if y_scores is not None:
        scores = np.asarray(y_scores, dtype=float).reshape(-1)
        try:
            res["roc_auc"] = float(roc_auc_score(y_true, scores) * 100.0)
        except Exception:
            res["roc_auc"] = 0.0
        try:
            res["pr_auc"] = float(average_precision_score(y_true, scores) * 100.0)
        except Exception:
            res["pr_auc"] = 0.0

    return res


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Find the score threshold that maximizes validation balanced accuracy."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)

    # Evaluate unique candidate thresholds
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = y_true[sorted_indices]

    n_pos = np.sum(sorted_labels == 1)
    n_neg = len(sorted_labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Scan unique thresholds
    unique_thresholds = np.unique(sorted_scores)
    if len(unique_thresholds) > 1000:
        # Sample quantiles if too many unique values
        unique_thresholds = np.quantile(sorted_scores, np.linspace(0.01, 0.99, 500))

    best_thresh = 0.0
    best_ba = -1.0

    for thresh in unique_thresholds:
        preds = (scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))
        tpr = tp / n_pos
        tnr = tn / n_neg
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba:
            best_ba = ba
            best_thresh = float(thresh)

    return best_thresh

