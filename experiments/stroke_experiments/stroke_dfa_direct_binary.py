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
from utils.seed import set_seed


def train_direct_binary(
    representation_path,
    seed
):
    set_seed(seed)

    data = load_representation(
        representation_path
    )

    H_train = data["H_train"]
    H_val = data["H_val"]
    H_test = data["H_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    positive_mask = y_train == 1
    negative_mask = y_train == 0

    positive_mean = H_train[
        positive_mask
    ].mean(dim=0)

    negative_mean = H_train[
        negative_mask
    ].mean(dim=0)

    direction = (
        positive_mean
        - negative_mean
    )

    binary_weights = torch.sign(
        direction
    )

    binary_weights[
        binary_weights == 0
    ] = 1

    val_scores = (
        H_val @ binary_weights
    )

    threshold, val_ba = find_best_threshold(
        y_val.numpy(),
        val_scores.numpy()
    )

    test_scores = (
        H_test @ binary_weights
    )

    metrics = calculate_metrics(
        y_test.numpy(),
        test_scores.numpy(),
        threshold
    )

    return {
        "method": "Direct Binary",
        "seed": int(data["seed"]),
        "head_seed": seed,
        "threshold": threshold,
        "validation_balanced_accuracy": val_ba,
        "binary_weights": binary_weights.numpy(),
        "bias": 0.0,
        **metrics
    }


if __name__ == "__main__":
    seed = 42

    representation_path = (
        f"data/stroke_dfa_representation_seed_{seed}.pt"
    )

    result = train_direct_binary(
        representation_path,
        seed
    )

    print("\nDirect Binary")

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