import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

import torch
import neal

from utils.stroke_representation import load_representation
from utils.stroke_metrics import (
    find_best_threshold,
    calculate_metrics
)
from utils.seed import set_seed
from utils.stroke_config import QUBO_NUM_READS


def build_ising_model(
    H,
    target,
    alpha
):
    H = H.float()
    target = target.float()

    H_centered = (
        H - H.mean(
            dim=0,
            keepdim=True
        )
    )

    target_centered = (
        target - target.mean()
    )

    gram = (
        H_centered.T
        @ H_centered
    )

    quadratic = (
        alpha ** 2
        * gram
    )

    linear = (
        -2.0
        * alpha
        * (
            H_centered.T
            @ target_centered
        )
    )

    hidden_size = H.shape[1]

    h = {}

    J = {}

    for i in range(hidden_size):
        h[i] = float(
            linear[i].item()
        )

    for i in range(hidden_size):
        for j in range(i + 1, hidden_size):
            coefficient = (
                2.0
                * quadratic[i, j]
            ).item()

            if abs(coefficient) > 1e-12:
                J[(i, j)] = float(
                    coefficient
                )

    return h, J


def optimize_binary_class(
    H,
    target,
    alpha,
    num_reads,
    seed
):
    h, J = build_ising_model(
        H,
        target,
        alpha
    )

    sampler = (
        neal.SimulatedAnnealingSampler()
    )

    response = sampler.sample_ising(
        h,
        J,
        num_reads=num_reads,
        seed=seed
    )

    best_sample = response.first.sample

    weights = torch.tensor(
        [
            1.0
            if best_sample[i] == 1
            else -1.0
            for i in range(H.shape[1])
        ],
        dtype=torch.float32
    )

    return weights


def calculate_alpha(
    H,
    labels
):
    H = H.float()

    H_augmented = torch.cat(
        [
            H,
            torch.ones(
                H.shape[0],
                1
            )
        ],
        dim=1
    )

    Y = torch.nn.functional.one_hot(
        labels.long(),
        num_classes=2
    ).float()

    W_ls = torch.linalg.lstsq(
        H_augmented,
        Y
    ).solution

    W_ls_positive = W_ls[
        :H.shape[1],
        1
    ]

    alpha = (
        torch.linalg.vector_norm(
            W_ls_positive
        )
        / (
            H.shape[1] ** 0.5
        )
    )

    return alpha.item()


def recover_bias(
    H,
    target,
    weights,
    alpha
):
    return (
        target.mean()
        - alpha
        * (
            H.mean(dim=0)
            @ weights
        )
    ).item()


def train_qubo(
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

    alpha = calculate_alpha(
        H_train,
        y_train
    )

    weights = optimize_binary_class(
        H=H_train,
        target=target_train,
        alpha=alpha,
        num_reads=QUBO_NUM_READS,
        seed=seed
    )

    bias = recover_bias(
        H_train,
        target_train,
        weights,
        alpha
    )

    val_scores = (
        alpha
        * (
            H_val @ weights
        )
        + bias
    )

    threshold, val_ba = find_best_threshold(
        y_val.numpy(),
        val_scores.numpy()
    )

    test_scores = (
        alpha
        * (
            H_test @ weights
        )
        + bias
    )

    metrics = calculate_metrics(
        y_test.numpy(),
        test_scores.numpy(),
        threshold
    )

    return {
        "method": "QUBO Binary",
        "seed": int(data["seed"]),
        "head_seed": seed,
        "threshold": threshold,
        "validation_balanced_accuracy": val_ba,
        "binary_weights": weights.numpy(),
        "alpha": float(alpha),
        "bias": float(bias),
        "qubo_num_reads": QUBO_NUM_READS,
        **metrics
    }


if __name__ == "__main__":
    seed = 42

    representation_path = (
        f"data/stroke_dfa_representation_seed_{seed}.pt"
    )

    result = train_qubo(
        representation_path,
        seed
    )

    print("\nQUBO Binary")

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
        f"Alpha: "
        f"{result['alpha']:.6f}"
    )

    print(
        f"Bias: "
        f"{result['bias']:.6f}"
    )

    print(
        f"QUBO reads: "
        f"{result['qubo_num_reads']}"
    )