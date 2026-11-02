"""Stroke Binarized-LS head with legacy and stable Ridge-LS variants.

The default uses float64 Ridge LS -> sign-binarized weights -> refitted
scale and intercept. The historical NumPy LS path remains available with
--ls-solver legacy. Scale/bias are fitted on training representations only;
validation chooses the score threshold, and test data are reserved for final metrics.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from models.ls_heads import fit_ridge_regression
from utils.stroke_binary_features import standardize_hidden_features
from utils.stroke_representation import load_stroke_head_representation, build_in_memory_representation
from utils.stroke_data import DATA_DIR
from utils.stroke_metrics import find_best_threshold, calculate_metrics
from experiments.stroke_experiments.stroke_bnn_dfa import train_dfa
from experiments.stroke_experiments.stroke_results_utils import append_result, Timer


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
    if denominator <= 1e-12:
        scale = 1.0
        bias = y_mean
    else:
        scale = float((z_centered @ y_centered) / denominator)
        bias = float(y_mean - scale * z_mean)
    return binary_weights, scale, bias, weights


def train_binarized_ls(representation_path, ls_solver="ridge", lambda_relative=1e-4):
    data = load_stroke_head_representation(representation_path)
    if ls_solver not in {"ridge", "legacy"}:
        raise ValueError("ls_solver must be 'ridge' or 'legacy'")

    if ls_solver == "ridge":
        H_train, H_val, H_test, transform = standardize_hidden_features(
            data["H_train"], data["H_val"], data["H_test"]
        )
        binary_weights, scale, bias, real_weights = fit_binarized_ridge_ls(
            H_train,
            data["y_train"],
            lambda_relative=lambda_relative,
        )
        method_name = "Binarized Ridge LS (refit scale+bias)"
        val_scores = (H_val.numpy() @ binary_weights) * scale + bias
        test_scores = (H_test.numpy() @ binary_weights) * scale + bias
        metadata = {
            "ls_solver": "float64_augmented_svd_ridge",
            "lambda_relative": float(lambda_relative),
            "scale": float(scale),
            "bias_treatment": "training-only refitted scalar scale and intercept",
            "hidden_feature_transform": transform["transform"],
            "standardized_constant_units": transform["constant_units"],
        }
    else:
        # Legacy path: preserve historical unregularized NumPy LS behavior
        H_train = data["H_train"]
        H_val = data["H_val"].numpy()
        H_test = data["H_test"].numpy()
        transform = {"transform": "raw_unscaled_hidden", "constant_units": []}
        binary_weights, scale, bias, real_weights = fit_legacy_binarized_ls(
            H_train, data["y_train"]
        )
        method_name = "Binarized LS (legacy)"
        val_scores = H_val @ binary_weights + bias
        test_scores = H_test @ binary_weights + bias
        metadata = {
            "ls_solver": "legacy_numpy_lstsq",
            "lambda_relative": "",
            "scale": float(scale),
            "bias_treatment": "continuous legacy LS intercept retained",
            "hidden_feature_transform": transform["transform"],
            "standardized_constant_units": transform["constant_units"],
        }

    threshold, val_ba = find_best_threshold(data["y_val"].numpy() if hasattr(data["y_val"], "numpy") else np.asarray(data["y_val"]), val_scores)
    y_test_np = data["y_test"].numpy() if hasattr(data["y_test"], "numpy") else np.asarray(data["y_test"])
    metrics = calculate_metrics(y_test_np, test_scores, threshold)

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
        default=None,
        help="Optional path to existing representation. If not supplied, backbone is trained fresh in memory."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for backbone training")
    parser.add_argument("--lambda-relative", type=float, default=1e-4)
    args = parser.parse_args()

    backbone_time = 0.0
    backbone_params = 1408
    if args.representation is not None and args.representation.exists():
        rep_source = args.representation
    else:
        print(f"Training fresh BNN-DFA backbone in memory with seed {args.seed}...")
        with Timer() as b_timer:
            dfa_result = train_dfa(args.seed, overwrite=True, output_dir=None)
        backbone_time = b_timer.seconds
        backbone_params = sum(p.numel() for p in dfa_result["model"].parameters())
        rep_source = build_in_memory_representation(dfa_result, feature_names=dfa_result["feature_names"])

    with Timer() as head_timer:
        result = train_binarized_ls(
            rep_source,
            ls_solver="ridge",
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

    total_time = backbone_time + head_timer.seconds
    head_params = 66  # 64 binary weights + scale + bias
    append_result(
        experiment="stroke_dfa_binarized_ls",
        method="BNN + DFA + Binarized Ridge LS Head (refit scale+bias)",
        seed=result["seed"],
        balanced_accuracy=result["balanced_accuracy"],
        f1_score=result["f1"],
        precision=result["precision"],
        recall=result["recall"],
        roc_auc=result["roc_auc"],
        pr_auc=result["pr_auc"],
        validation_balanced_accuracy=result["validation_balanced_accuracy"],
        threshold=result["threshold"],
        training_time_sec=backbone_time,
        backbone_training_time_sec=backbone_time,
        head_fit_time_sec=head_timer.seconds,
        current_run_total_time_sec=total_time,
        end_to_end_training_time_sec=total_time,
        parameter_count=backbone_params,
        backbone_parameter_count=backbone_params,
        head_parameter_count=head_params,
        total_parameter_count=backbone_params + head_params,
    )


if __name__ == "__main__":
    main()
