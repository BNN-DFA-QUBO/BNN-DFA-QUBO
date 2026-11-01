"""CIFAR-10 BNN+DFA with Straight-Through Estimator (STE) Binary Head.

Directly trains {-1, +1} binary head using STE and retains the checkpoint
with the highest validation accuracy (validation-only checkpointing).
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
from utils.CIFAR10.data import get_device
from utils.CIFAR10.representation import DEFAULT_REPRESENTATION_DIR
from utils.MNIST.seed import set_seed


def binary_ste(w: torch.Tensor) -> torch.Tensor:
    """Hard {-1, +1} in forward pass, identity gradient in backward pass."""
    binary = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    return binary.detach() - w.detach() + w


def train_ste_with_validation(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    H_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.01,
    seed: int | None = None,
) -> tuple[dict[str, Any], float]:
    """Train STE head and retain the checkpoint with highest validation accuracy."""
    if seed is not None:
        torch.manual_seed(seed)

    H_train, H_val = H_train.float(), H_val.float()
    y_train, y_val = y_train.long(), y_val.long()
    hidden_size = H_train.shape[1]
    num_classes = 10

    latent = (torch.randn(hidden_size, num_classes) * 0.01).requires_grad_()
    bias = torch.zeros(num_classes, requires_grad=True)
    optimizer = torch.optim.Adam([latent, bias], lr=lr)

    best = None
    t0 = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = H_train @ binary_ste(latent) + bias
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_logits = H_val @ binary_ste(latent) + bias
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item() * 100.0

            if best is None or val_acc > best["val_accuracy"]:
                hard_w = torch.where(
                    latent >= 0, torch.ones_like(latent), -torch.ones_like(latent)
                ).detach().clone()
                train_logits = H_train @ hard_w + bias
                train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item() * 100.0
                best = {
                    "val_accuracy": val_acc,
                    "train_accuracy": train_acc,
                    "epoch": epoch + 1,
                    "weights": hard_w,
                    "bias": bias.detach().clone(),
                }

    fit_time = time.perf_counter() - t0
    return best, fit_time


def evaluate_ste_head(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    H_val: torch.Tensor,
    y_val: torch.Tensor,
    H_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.01,
    seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, float], float]:
    best, fit_time = train_ste_with_validation(
        H_train, y_train, H_val, y_val, epochs=epochs, lr=lr, seed=seed
    )
    with torch.no_grad():
        test_logits = H_test.float() @ best["weights"] + best["bias"]
        test_acc = (test_logits.argmax(dim=1) == y_test.long()).float().mean().item() * 100.0

    accuracies = {
        "train": best["train_accuracy"],
        "validation": best["val_accuracy"],
        "test": test_acc,
    }
    return best, accuracies, fit_time


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 BNN+DFA STE Binary head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ste-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
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

    best, accs, fit_time = evaluate_ste_head(
        H_train, y_train, H_val, y_val, H_test, y_test,
        epochs=args.ste_epochs, lr=args.lr, seed=args.seed,
    )

    print("\nSTE Binary Head Results (best validation epoch):")
    print(f"Selected Epoch:      {best['epoch']}/{args.ste_epochs}")
    print(f"Train Accuracy:      {accs['train']:.2f}%")
    print(f"Validation Accuracy: {accs['validation']:.2f}%")
    print(f"Test Accuracy:       {accs['test']:.2f}%")
    print(f"Fit time:            {fit_time:.4f}s")

    metadata = representation["metadata"]
    backbone_params = metadata.get("backbone_parameter_count", 29002)
    head_params = int(best["weights"].numel() + best["bias"].numel())  # 1280 + 10 = 1290

    append_result(
        experiment="cifar_dfa_ste_binary",
        method="STE Binary",
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
        notes=f"selected_epoch={best['epoch']}",
    )


if __name__ == "__main__":
    main()
