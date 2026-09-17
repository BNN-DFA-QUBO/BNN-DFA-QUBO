import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

import numpy as np
import torch

from utils.stroke_representation import load_representation
from utils.stroke_metrics import (
    find_best_threshold,
    calculate_metrics
)


def train_real_ls(representation_path):
    data = load_representation(
        representation_path
    )

    H_train = data["H_train"].numpy()
    H_val = data["H_val"].numpy()
    H_test = data["H_test"].numpy()

    y_train = data["y_train"].numpy()
    y_val = data["y_val"].numpy()
    y_test = data["y_test"].numpy()

    H_train_aug = np.column_stack(
        [
            H_train,
            np.ones(len(H_train))
        ]
    )

    H_val_aug = np.column_stack(
        [
            H_val,
            np.ones(len(H_val))
        ]
    )

    H_test_aug = np.column_stack(
        [
            H_test,
            np.ones(len(H_test))
        ]
    )

    weights = np.linalg.lstsq(
        H_train_aug,
        y_train,
        rcond=None
    )[0]

    val_scores = H_val_aug @ weights

    threshold, val_ba = find_best_threshold(
        y_val,
        val_scores
    )

    test_scores = H_test_aug @ weights

    metrics = calculate_metrics(
        y_test,
        test_scores,
        threshold
    )

    return {
        "method": "Real LS",
        "seed": int(data["seed"]),
        "threshold": threshold,
        "validation_balanced_accuracy": val_ba,
        **metrics
    }


if __name__ == "__main__":
    representation_path = (
        "data/stroke_dfa_representation_seed_42.pt"
    )

    result = train_real_ls(
        representation_path
    )

    print("\nReal LS")
    print(
        f"Validation BA: "
        f"{result['validation_balanced_accuracy'] * 100:.4f}%"
    )

    print(
        f"Test BA: "
        f"{result['balanced_accuracy'] * 100:.4f}%"
    )

    print(
        f"Precision: "
        f"{result['precision'] * 100:.4f}%"
    )

    print(
        f"Recall: "
        f"{result['recall'] * 100:.4f}%"
    )

    print(
        f"F1: "
        f"{result['f1'] * 100:.4f}%"
    )

    print(
        f"ROC-AUC: "
        f"{result['roc_auc'] * 100:.4f}%"
    )

    print(
        f"PR-AUC: "
        f"{result['pr_auc'] * 100:.4f}%"
    )

    print(
        f"Threshold: "
        f"{result['threshold']:.6f}"
    )