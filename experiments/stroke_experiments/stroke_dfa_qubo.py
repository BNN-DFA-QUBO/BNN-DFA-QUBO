import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

import torch
import neal

from models.ls_heads import fit_ridge_regression

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
    labels,
    lambda_relative=1e-4,
):
    """Estimate a QUBO score scale from a stable signed-target Ridge fit.

    The QUBO energy is written for targets in {-1,+1}, so scale calibration
    uses that same encoding. The shared CPU float64 SVD Ridge solver is more
    robust than an unregularized float32 multi-output solve on rank-deficient
    hidden representations.
    """
    H64 = torch.as_tensor(H).detach().to(device="cpu", dtype=torch.float64)
    labels64 = torch.as_tensor(labels).detach().reshape(-1).to(
        device="cpu", dtype=torch.float64
    )
    if H64.ndim != 2 or H64.shape[0] != labels64.shape[0] or H64.shape[0] == 0:
        raise ValueError("QUBO scale calibration requires aligned non-empty hidden features and labels")
    if not torch.isfinite(H64).all() or not torch.isfinite(labels64).all():
        raise ValueError("QUBO scale calibration requires finite features and labels")
    if not torch.isin(labels64, torch.tensor([0.0, 1.0], dtype=torch.float64)).all():
        raise ValueError("QUBO scale calibration expects binary labels encoded as {0,1}")

    signed_targets = 2.0 * labels64 - 1.0
    coefficients = fit_ridge_regression(
        H64, signed_targets, lambda_relative=lambda_relative
    )[:-1, 0]
    alpha = torch.linalg.vector_norm(coefficients) / (H64.shape[1] ** 0.5)
    alpha_value = float(alpha.item())
    if not torch.isfinite(alpha) or alpha_value <= 1e-8:
        raise ValueError(
            "QUBO scale calibration is degenerate (alpha is zero or near zero). "
            "Inspect the saved representation for collapsed or non-informative hidden features."
        )
    return alpha_value


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

    from utils.stroke_data import DATA_DIR
    representation_path = DATA_DIR / f"stroke_dfa_representation_seed_{seed}.pt"

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
