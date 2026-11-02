import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

import argparse
import numpy as np
import torch

from utils.stroke_representation import load_stroke_head_representation, build_in_memory_representation
from utils.stroke_binary_features import standardize_hidden_features
from utils.stroke_metrics import (
    find_best_threshold,
    calculate_metrics
)
from utils.seed import set_seed
from experiments.stroke_experiments.stroke_bnn_dfa import train_dfa
from experiments.stroke_experiments.stroke_results_utils import append_result, Timer


def train_direct_binary(
    representation_path,
    seed
):
    set_seed(seed)

    data = load_stroke_head_representation(
        representation_path
    )

    H_train, H_val, H_test, transform = standardize_hidden_features(
        data["H_train"], data["H_val"], data["H_test"]
    )

    y_train = torch.as_tensor(data["y_train"]).float().reshape(-1)
    y_val = torch.as_tensor(data["y_val"]).float().reshape(-1)
    y_test = torch.as_tensor(data["y_test"]).float().reshape(-1)

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
        "hidden_feature_transform": transform["transform"],
        "standardized_constant_units": transform["constant_units"],
        **metrics
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stroke Direct Binary Head")
    parser.add_argument("--representation", type=Path, default=None,
                        help="Optional path to existing representation. If not supplied, backbone is trained fresh in memory.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for training")
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
        result = train_direct_binary(
            rep_source,
            args.seed
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

    total_time = backbone_time + head_timer.seconds
    head_params = 64  # 64 binary weights
    append_result(
        experiment="stroke_dfa_direct_binary",
        method="BNN + DFA + Direct Binary Head",
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
