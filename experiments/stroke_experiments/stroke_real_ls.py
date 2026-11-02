import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

import argparse
import numpy as np
import torch

from utils.stroke_representation import load_stroke_head_representation, build_in_memory_representation
from utils.stroke_data import DATA_DIR
from utils.stroke_metrics import (
    find_best_threshold,
    calculate_metrics
)
from experiments.stroke_experiments.stroke_bnn_dfa import train_dfa
from experiments.stroke_experiments.stroke_results_utils import append_result, Timer


def train_real_ls(representation_path):
    data = load_stroke_head_representation(
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
    parser = argparse.ArgumentParser(description="Stroke Real-Valued LS Head")
    parser.add_argument("--representation", type=Path, default=None,
                        help="Optional path to existing representation. If not supplied, backbone is trained fresh in memory.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for backbone training")
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
        result = train_real_ls(rep_source)

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

    total_time = backbone_time + head_timer.seconds
    head_params = 65
    append_result(
        experiment="stroke_real_ls",
        method="BNN + DFA + Ridge LS Head",
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
