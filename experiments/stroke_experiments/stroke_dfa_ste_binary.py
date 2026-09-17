import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

import torch

from utils.stroke_representation import load_representation
from utils.stroke_metrics import (
    find_best_threshold,
    calculate_metrics
)
from utils.seed import set_seed
from utils.stroke_config import (
    STE_EPOCHS,
    STE_LEARNING_RATE
)


def train_ste_binary(
    representation_path,
    seed
):
    set_seed(seed)

    data = load_representation(
        representation_path
    )

    H_train = data["H_train"].float()
    H_val = data["H_val"].float()
    H_test = data["H_test"].float()

    y_train = data["y_train"].float()
    y_val = data["y_val"].float()
    y_test = data["y_test"].float()

    target_train = (
        y_train * 2.0
        - 1.0
    )

    weights = torch.randn(
        H_train.shape[1]
    ) * 0.01

    bias = torch.tensor(
        0.0
    )

    weights.requires_grad_()
    bias.requires_grad_()

    optimizer = torch.optim.Adam(
        [weights, bias],
        lr=STE_LEARNING_RATE
    )

    for _ in range(STE_EPOCHS):
        optimizer.zero_grad()

        binary_weights = (
            weights.sign().detach()
            - weights.detach()
            + weights
        )

        logits = (
            H_train @ binary_weights
            + bias
        )

        loss = torch.mean(
            (
                logits
                - target_train
            ) ** 2
        )

        loss.backward()

        optimizer.step()

    with torch.no_grad():
        final_binary_weights = (
            weights.sign()
        )

        final_binary_weights[
            final_binary_weights == 0
        ] = 1

        val_scores = (
            H_val
            @ final_binary_weights
            + bias
        )

        test_scores = (
            H_test
            @ final_binary_weights
            + bias
        )

    threshold, val_ba = find_best_threshold(
        y_val.numpy(),
        val_scores.numpy()
    )

    metrics = calculate_metrics(
        y_test.numpy(),
        test_scores.numpy(),
        threshold
    )

    return {
        "method": "STE Binary",
        "seed": int(data["seed"]),
        "head_seed": seed,
        "threshold": threshold,
        "validation_balanced_accuracy": val_ba,
        "binary_weights": (
            final_binary_weights.numpy()
        ),
        "bias": float(
            bias.detach()
        ),
        **metrics
    }


if __name__ == "__main__":
    seed = 42

    representation_path = (
        f"data/stroke_dfa_representation_seed_{seed}.pt"
    )

    result = train_ste_binary(
        representation_path,
        seed
    )

    print("\nSTE Binary")

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

    print(
        f"Bias: "
        f"{result['bias']:.6f}"
    )