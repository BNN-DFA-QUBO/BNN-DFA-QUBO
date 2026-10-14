"""Metrics and validation-only decision-threshold selection for Stroke heads."""
import numpy as np

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def find_best_threshold(y_true, scores, thresholds=None):
    """Choose the threshold with highest balanced accuracy on validation data.

    By default, test every unique score boundary plus a boundary above the
    maximum score. This evaluates every distinct prediction set and avoids a
    fixed linear grid skipping useful thresholds when scores contain outliers
    or large gaps. Callers must pass validation labels/scores, not test labels.
    If an explicit threshold array is provided, it is respected for backwards
    compatibility.
    """
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(scores):
        raise ValueError("y_true and scores must be non-empty and have equal length")
    if not np.isfinite(scores).all():
        raise ValueError("Threshold selection requires finite scores")

    if thresholds is None:
        unique_scores = np.unique(scores)
        thresholds = np.concatenate(
            (unique_scores, [np.nextafter(unique_scores[-1], np.inf)])
        )
    else:
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

    return best_threshold, best_ba


def calculate_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(scores):
        raise ValueError("y_true and scores must be non-empty and have equal length")
    if not np.isfinite(scores).all():
        raise ValueError("Metric calculation requires finite scores")

    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive": int(predictions.sum()),
        "actual_positive": int(np.asarray(y_true).sum()),
        "threshold": float(threshold),
    }
