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
    """Choose the threshold maximizing validation balanced accuracy.

    The default path evaluates every unique score boundary in O(n log n) by
    sorting scores once and calculating cumulative class counts. This is
    exactly equivalent to trying each distinct prediction set, but avoids a
    Python-level metric call for every threshold (important when ranking many
    STE/QUBO candidates). Ties retain the lowest threshold, matching the
    former ascending-boundary loop. Explicit threshold grids retain their
    previous behavior.
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
        return best_threshold, best_ba

    n_positive = int(np.sum(y_true == 1))
    n_negative = int(np.sum(y_true == 0))
    # Preserve sklearn's behavior for single-class input; normal Stroke
    # validation/test splits are stratified and include both classes.
    if n_positive == 0 or n_negative == 0:
        unique_scores = np.unique(scores)
        candidates = np.concatenate(
            (unique_scores, [np.nextafter(unique_scores[-1], np.inf)])
        )
        best_threshold = float(candidates[0])
        best_ba = -1.0
        for threshold in candidates:
            predictions = (scores >= threshold).astype(int)
            ba = balanced_accuracy_score(y_true, predictions)
            if ba > best_ba:
                best_ba = float(ba)
                best_threshold = float(threshold)
        return best_threshold, best_ba

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y_true[order]
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    cumulative_positive = np.cumsum(sorted_y == 1)
    cumulative_negative = np.cumsum(sorted_y == 0)
    positive_below = np.r_[0, cumulative_positive[group_ends[:-1]]]
    negative_below = np.r_[0, cumulative_negative[group_ends[:-1]]]

    # At threshold t, predictions are positive iff score >= t. The examples
    # below t are false negatives / true negatives respectively.
    sensitivity = (n_positive - positive_below) / n_positive
    specificity = negative_below / n_negative
    balanced_accuracy = (sensitivity + specificity) / 2.0

    # Boundary above the maximum predicts every case as negative, BA=0.5.
    candidate_scores = np.r_[sorted_scores[group_ends], np.nextafter(sorted_scores[-1], np.inf)]
    candidate_ba = np.r_[balanced_accuracy, 0.5]
    best_index = int(np.argmax(candidate_ba))
    return float(candidate_scores[best_index]), float(candidate_ba[best_index])


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
