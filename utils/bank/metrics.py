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


def find_best_threshold(y_true: np.ndarray | list, scores: np.ndarray | list, thresholds: np.ndarray | list | None = None) -> float:
    """Find the exact score threshold that maximizes validation balanced accuracy in O(N log N).

    Evaluates every distinct decision boundary across all sorted score values,
    avoiding any heuristic down-sampling or quantile approximations.
    """
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(scores):
        raise ValueError("y_true and scores must be non-empty and have equal length")
    if not np.isfinite(scores).all():
        raise ValueError("Threshold selection requires finite scores")

    if thresholds is not None:
        thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if len(thresholds) == 0 or not np.isfinite(thresholds).all():
            raise ValueError("Explicit thresholds must be non-empty and finite")
        best_threshold = float(thresholds[0])
        best_ba = -1.0
        for threshold in thresholds:
            predictions = (scores >= threshold).astype(int)
            ba = balanced_accuracy_score(y_true, predictions)
            if ba > best_ba:
                best_ba = float(ba)
                best_threshold = float(threshold)
        return best_threshold

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y_true[order]
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]

    cumulative_positive = np.cumsum(sorted_y == 1)
    cumulative_negative = np.cumsum(sorted_y == 0)
    positive_below = np.r_[0, cumulative_positive[group_ends[:-1]]]
    negative_below = np.r_[0, cumulative_negative[group_ends[:-1]]]

    sensitivity = (n_pos - positive_below) / n_pos
    specificity = negative_below / n_neg
    balanced_accuracy = (sensitivity + specificity) / 2.0

    candidate_scores = np.r_[sorted_scores[group_ends], np.nextafter(sorted_scores[-1], np.inf)]
    candidate_ba = np.r_[balanced_accuracy, 0.5]
    best_index = int(np.argmax(candidate_ba))

    return float(candidate_scores[best_index])
