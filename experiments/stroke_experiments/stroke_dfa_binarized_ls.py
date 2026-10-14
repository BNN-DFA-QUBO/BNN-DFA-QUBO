"""Stroke Binarized-LS head with legacy and stable Ridge-LS variants.

Default: float64 Ridge LS -> sign-binarized feature weights -> refit scalar
scale and intercept using training representations only. Validation chooses
the final score threshold; test data are used only for metric reporting.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from models.ls_heads import fit_ridge_regression
from utils.stroke_representation import load_representation
from utils.stroke_data import DATA_DIR
from utils.stroke_metrics import find_best_threshold, calculate_metrics


def _numpy(value, dtype=np.float64):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def fit_legacy_binarized_ls(H_train, y_train):
    """Legacy behavior: unregularized NumPy LS, sign weights, retain intercept."""
    H = _numpy(H_train)
    y = _numpy(y_train).reshape(-1)
    design = np.column_stack((H, np.ones(len(H), dtype=np.float64)))
    weights, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    binary_weights = np.sign(weights[:-1])
    binary_weights[binary_weights == 0] = 1.0
    return binary_weights, 1.0, float(weights[-1]), weights


def fit_binarized_ridge_ls(H_train, y_train, lambda_relative=1e-4):
    """Fit stable Ridge LS and then refit binary-head scale and intercept."""
    H = _numpy(H_train)
    y = _numpy(y_train).reshape(-1)
    weights = fit_ridge_regression(
        torch.as_tensor(H, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        lambda_relative=lambda_relative,
    )[:, 0].numpy()

    binary_weights = np.sign(weights[:-1])
    binary_weights[binary_weights == 0] = 1.0

    # Refit z * scale + bias to the training-only {0,1} targets.
    z = H @ binary_weights
    z_mean = float(z.mean())
    y_mean = float(y.mean())
    z_centered = z - z_mean
    y_centered = y - y_mean
    denominator = float(z_centered @ z_centered)
    scale_raw = (
        float(z_centered @ y_centered) / denominator
        if denominator > 1e-12 else 0.0
    )

    # Ensure a non-negative scale while preserving the fitted train scores.
    if scale_raw < 0:
        binary_weights *= -1.0
        scale = -scale_raw
    else:
        scale = scale_raw
    bias = y_mean - scale_raw * z_mean
    return binary_weights, float(scale), float(bias), weights


def train_binarized_ls(representation_path, ls_solver="legacy", lambda_relative=1e-4):
    """Evaluate one selected binarized LS variant on a saved representation."""
    if ls_solver not in {"ridge", "legacy"}:
        raise ValueError("ls_solver must be 'ridge' or 'legacy'")
    if not np.isfinite(lambda_relative) or lambda_relative < 0:
        raise ValueError("lambda_relative must be finite and >= 0")

    data = load_representation(representation_path)
    H_train = _numpy(data["H_train"])
    H_val = _numpy(data["H_val"])
    H_test = _numpy(data["H_test"])
    y_train = _numpy(data["y_train"]).reshape(-1)
    y_val = _numpy(data["y_val"]).reshape(-1)
    y_test = _numpy(data["y_test"]).reshape(-1)

    if ls_solver == "ridge":
        binary_weights, scale, bias, real_weights = fit_binarized_ridge_ls(
            H_train, y_train, lambda_relative=lambda_relative
        )
        method_name = "Binarized Ridge LS (refit scale+bias)"
        val_scores = (H_val @ binary_weights) * scale + bias
        test_scores = (H_test @ binary_weights) * scale + bias
        metadata = {
            "ls_solver": "float64_augmented_svd_ridge",
            "lambda_relative": float(lambda_relative),
            "scale": float(scale),
            "bias_treatment": "refitted using training rows after sign binarization",
        }
    else:
        binary_weights, scale, bias, real_weights = fit_legacy_binarized_ls(
            H_train, y_train
        )
        method_name = "Binarized LS (legacy)"
        val_scores = H_val @ binary_weights + bias
        test_scores = H_test @ binary_weights + bias
        metadata = {
            "ls_solver": "legacy_numpy_lstsq",
            "lambda_relative": "",
            "scale": float(scale),
            "bias_treatment": "continuous legacy LS intercept retained",
        }

    threshold, val_ba = find_best_threshold(y_val, val_scores)
    metrics = calculate_metrics(y_test, test_scores, threshold)

    return {
        "method": method_name,
        "seed": int(data.get("training_seed", data.get("seed", -1))),
        "representation_seed": int(data.get("training_seed", data.get("seed", -1))),
        "split_seed": int(data.get("split_seed", -1)),
        "threshold": float(threshold),
        "validation_balanced_accuracy": float(val_ba),
        "binary_weights": binary_weights,
        "real_weights": real_weights,
        "bias": float(bias),
        **metadata,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation",
        type=Path,
        default=DATA_DIR / "stroke_dfa_representation_seed_42.pt",
    )
    parser.add_argument("--ls-solver", choices=("ridge", "legacy"), default="legacy",
                        help="ridge: stable Ridge-LS weights with refitted scale/bias; legacy: existing NumPy LS head")
    parser.add_argument("--lambda-relative", type=float, default=1e-4)
    args = parser.parse_args()

    result = train_binarized_ls(
        args.representation,
        ls_solver=args.ls_solver,
        lambda_relative=args.lambda_relative,
    )

    print(f"\n{result['method']}")
    print(f"LS solver: {result['ls_solver']}")
    print(f"Validation BA: {result['validation_balanced_accuracy'] * 100:.4f}%")
    print(f"Test BA: {result['balanced_accuracy'] * 100:.4f}%")
    print(f"Precision: {result['precision'] * 100:.4f}%")
    print(f"Recall: {result['recall'] * 100:.4f}%")
    print(f"F1: {result['f1'] * 100:.4f}%")
    print(f"ROC-AUC: {result['roc_auc'] * 100:.4f}%")
    print(f"PR-AUC: {result['pr_auc'] * 100:.4f}%")
    print(f"Threshold: {result['threshold']:.6f}")


if __name__ == "__main__":
    main()
