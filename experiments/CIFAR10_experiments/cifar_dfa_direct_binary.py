"""CIFAR-10 BNN+DFA with Direct Binary Head.

Optimizes latent continuous weights using tanh relaxation, then projects
to hard {-1, +1} binary weights.
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


def optimize_direct_binary_head(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int = 10,
    steps: int = 200,
    lr: float = 0.05,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Train direct binary head with tanh relaxation and project to hard signs."""
    if seed is not None:
        torch.manual_seed(seed)

    H = H_train.float()
    hidden_size = H.shape[1]

    latent = (torch.randn(hidden_size, num_classes, dtype=torch.float32) * 0.01).requires_grad_()
    bias = torch.zeros(num_classes, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([latent, bias], lr=lr)

    t0 = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad()
        w_relaxed = torch.tanh(latent)
        logits = H @ w_relaxed + bias
        loss = F.cross_entropy(logits, y_train.long())
        loss.backward()
        optimizer.step()
    fit_time = time.perf_counter() - t0

    with torch.no_grad():
        w_binary = torch.where(latent >= 0, torch.ones_like(latent), -torch.ones_like(latent))
        final_bias = bias.detach().clone()

    return w_binary, final_bias, fit_time


def predict_direct_binary(H: torch.Tensor, W: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return H.float() @ W.float() + bias.float()


def evaluate_direct_binary_head(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    H_val: torch.Tensor,
    y_val: torch.Tensor,
    H_test: torch.Tensor,
    y_test: torch.Tensor,
    steps: int = 200,
    lr: float = 0.05,
    seed: int | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, float], float]:
    w_bin, bias, fit_time = optimize_direct_binary_head(
        H_train, y_train, num_classes=10, steps=steps, lr=lr, seed=seed
    )

    def acc(H, y):
        scores = predict_direct_binary(H, w_bin, bias)
        return float((scores.argmax(dim=1) == y).float().mean().item() * 100.0)

    accuracies = {
        "train": acc(H_train, y_train),
        "validation": acc(H_val, y_val),
        "test": acc(H_test, y_test),
    }
    return (w_bin, bias), accuracies, fit_time


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 BNN+DFA Direct Binary head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
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

    (w_bin, bias), accs, fit_time = evaluate_direct_binary_head(
        H_train, y_train, H_val, y_val, H_test, y_test,
        steps=args.steps, lr=args.lr, seed=args.seed,
    )

    print("\nDirect Binary Head Results:")
    print(f"Train Accuracy:      {accs['train']:.2f}%")
    print(f"Validation Accuracy: {accs['validation']:.2f}%")
    print(f"Test Accuracy:       {accs['test']:.2f}%")
    print(f"Fit time:            {fit_time:.4f}s")

    metadata = representation["metadata"]
    backbone_params = metadata.get("backbone_parameter_count", 29002)
    head_params = int(w_bin.numel() + bias.numel())  # 1280 + 10 = 1290

    append_result(
        experiment="cifar_dfa_direct_binary",
        method="Direct Binary",
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
