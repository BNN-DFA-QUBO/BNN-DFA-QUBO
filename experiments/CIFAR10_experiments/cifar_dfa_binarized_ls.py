"""CIFAR-10 BNN+DFA with Binarized Least-Squares Head.

Binarizes the continuous Ridge-LS weights and refits class-specific scales
and biases on the training representation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from experiments.CIFAR10_experiments.cifar_representation import (
    load_or_train_cifar10_representation,
)
from experiments.CIFAR10_experiments.results_utils import (
    append_result,
)
from models.ls_heads import fit_ridge_classifier
from utils.CIFAR10.data import get_device
from utils.CIFAR10.representation import DEFAULT_REPRESENTATION_DIR
from utils.MNIST.seed import set_seed


def fit_binarized_ridge(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    ridge_weights: torch.Tensor,
    num_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bipolarize Ridge features (zero maps to +1) and refit scale & bias on training rows."""
    classifier = ridge_weights.detach().to(device="cpu", dtype=torch.float32)
    feature_weights = classifier[:-1, :]
    binary_weights = torch.where(feature_weights >= 0, 1.0, -1.0)
    targets = F.one_hot(y_train.to(torch.long), num_classes=num_classes).float()
    scales, biases = [], []

    for class_id in range(num_classes):
        z = H_train.float() @ binary_weights[:, class_id]
        target = targets[:, class_id]
        z_mean, target_mean = z.mean(), target.mean()
        z_centered, target_centered = z - z_mean, target - target_mean
        denominator = z_centered @ z_centered
        if denominator.item() > 1e-12:
            scale_raw = (z_centered @ target_centered) / denominator
        else:
            scale_raw = torch.tensor(1.0, dtype=z.dtype)

        if scale_raw.item() < 0:
            binary_weights[:, class_id] *= -1
            scale = -scale_raw
        else:
            scale = scale_raw
        scales.append(scale)
        biases.append(target_mean - scale_raw * z_mean)

    return binary_weights, torch.stack(scales), torch.stack(biases)


def predict_binary_scores(
    H: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
) -> torch.Tensor:
    """Compute logits: H @ (W * scale) + bias."""
    return H.float() @ (weights * scales.unsqueeze(0)) + biases.unsqueeze(0)


def evaluate_binarized_head(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    H_val: torch.Tensor,
    y_val: torch.Tensor,
    H_test: torch.Tensor,
    y_test: torch.Tensor,
    ridge_weights: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, float], float]:
    t0 = time.perf_counter()
    bin_w, scales, biases = fit_binarized_ridge(H_train, y_train, ridge_weights)
    fit_time = time.perf_counter() - t0

    def acc(H, y):
        scores = predict_binary_scores(H, bin_w, scales, biases)
        return float((scores.argmax(dim=1) == y).float().mean().item() * 100.0)

    accuracies = {
        "train": acc(H_train, y_train),
        "validation": acc(H_val, y_val),
        "test": acc(H_test, y_test),
    }
    return (bin_w, scales, biases), accuracies, fit_time


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 BNN+DFA Binarized Ridge LS head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--representation-dir",
        type=Path,
        default=DEFAULT_REPRESENTATION_DIR,
    )
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

    ridge_weights = fit_ridge_classifier(H_train, y_train, num_classes=10)
    (bin_w, scales, biases), accs, fit_time = evaluate_binarized_head(
        H_train, y_train, H_val, y_val, H_test, y_test, ridge_weights
    )

    print("\nBinarized Ridge LS Results:")
    print(f"Train Accuracy:      {accs['train']:.2f}%")
    print(f"Validation Accuracy: {accs['validation']:.2f}%")
    print(f"Test Accuracy:       {accs['test']:.2f}%")
    print(f"Fit time:            {fit_time:.4f}s")

    metadata = representation["metadata"]
    backbone_params = metadata.get("backbone_parameter_count", 29002)
    head_params = int(bin_w.numel() + scales.numel() + biases.numel())  # 1280 + 10 + 10 = 1300

    append_result(
        experiment="cifar_dfa_binarized_ls",
        method="Binarized Least Squares",
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
