"""CIFAR-10 BNN+DFA with Real-Valued Least-Squares (Ridge LS) Head.

Uses the numerically stable float64 Ridge solver from models/ls_heads.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import torch

from experiments.CIFAR10_experiments.cifar_representation import (
    load_or_train_cifar10_representation,
)
from experiments.CIFAR10_experiments.results_utils import (
    append_result,
    count_parameters,
)
from models.ls_heads import (
    accuracy_from_scores,
    fit_ridge_classifier,
    predict_scores,
)
from utils.CIFAR10.data import get_device
from utils.CIFAR10.representation import DEFAULT_REPRESENTATION_DIR
from utils.MNIST.seed import set_seed


def fit_and_evaluate_ridge(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    H_val: torch.Tensor,
    y_val: torch.Tensor,
    H_test: torch.Tensor,
    y_test: torch.Tensor,
    lambda_relative: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float], float]:
    """Fit Ridge LS head on training representations and evaluate all splits."""
    t0 = time.perf_counter()
    w = fit_ridge_classifier(
        H=H_train,
        labels=y_train,
        num_classes=10,
        lambda_relative=lambda_relative,
    )
    fit_time = time.perf_counter() - t0

    train_scores = predict_scores(H_train, w)
    val_scores = predict_scores(H_val, w)
    test_scores = predict_scores(H_test, w)

    accuracies = {
        "train": accuracy_from_scores(train_scores, y_train),
        "validation": accuracy_from_scores(val_scores, y_val),
        "test": accuracy_from_scores(test_scores, y_test),
    }
    return w, accuracies, fit_time


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 BNN+DFA Ridge LS head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--representation-dir",
        type=Path,
        default=DEFAULT_REPRESENTATION_DIR,
    )
    parser.add_argument("--lambda-relative", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Seed={args.seed} | Device={device}")

    representation, rep_path, rep_source = load_or_train_cifar10_representation(
        seed=args.seed,
        epochs=args.epochs,
        validation_fraction=args.validation_fraction,
        device=device,
        output_dir=args.representation_dir,
    )
    H_train, y_train = representation["H_train"], representation["y_train"]
    H_val, y_val = representation["H_validation"], representation["y_validation"]
    H_test, y_test = representation["H_test"], representation["y_test"]

    w, accs, fit_time = fit_and_evaluate_ridge(
        H_train, y_train, H_val, y_val, H_test, y_test, args.lambda_relative
    )

    print(f"\nRidge LS Results (lambda={args.lambda_relative}):")
    print(f"Train Accuracy:      {accs['train']:.2f}%")
    print(f"Validation Accuracy: {accs['validation']:.2f}%")
    print(f"Test Accuracy:       {accs['test']:.2f}%")
    print(f"Fit time:            {fit_time:.4f}s")

    metadata = representation["metadata"]
    backbone_params = metadata.get("backbone_parameter_count", 29002)
    head_params = int(w.numel())  # 129 * 10 = 1290

    append_result(
        experiment="cifar_dfa_least_squares",
        method="Real Least Squares (Ridge LS)",
        seed=args.seed,
        test_accuracy=accs["test"],
        validation_accuracy=accs["validation"],
        training_accuracy=accs["train"],
        parameter_count=backbone_params + head_params,
        backbone_training_time_sec=representation.get("training_seconds", 0.0),
        feature_extraction_time_sec=representation.get("feature_extraction_seconds", 0.0),
        head_fit_time_sec=fit_time,
        backbone_parameter_count=backbone_params,
        head_parameter_count=head_params,
        total_parameter_count=backbone_params + head_params,
        pipeline_timing_status="complete",
        representation_artifact_id=representation["artifact_id"],
        representation_path=str(rep_path.resolve()),
        representation_cache_status=rep_source,
        n_train=len(y_train),
        n_validation=len(y_val),
        n_test=len(y_test),
        dataset_fingerprint=metadata.get("dataset_fingerprint", ""),
        split_protocol=metadata.get("split_protocol", ""),
        representation_sha256=representation.get("representation_sha256", ""),
    )


if __name__ == "__main__":
    main()
