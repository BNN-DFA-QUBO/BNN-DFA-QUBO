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


def find_best_threshold(
    y_true,
    scores,
    thresholds=None
):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if thresholds is None:
        thresholds = np.linspace(
            scores.min(),
            scores.max(),
            1000
        )

    best_threshold = thresholds[0]
    best_ba = -1.0

    for threshold in thresholds:
        predictions = (
            scores >= threshold
        ).astype(int)

        ba = balanced_accuracy_score(
            y_true,
            predictions
        )

        if ba > best_ba:
            best_ba = ba
            best_threshold = threshold

    return best_threshold, best_ba


def calculate_metrics(
    y_true,
    scores,
    threshold
):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    predictions = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            scores
        ),
        "pr_auc": average_precision_score(
            y_true,
            scores
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive": int(predictions.sum()),
        "actual_positive": int(y_true.sum()),
        "threshold": float(threshold)
    }