"""Train CIFAR-10 BNN backbone with Direct Feedback Alignment (DFA).

Standalone diagnostic runner and reusable training/feature-collection routines.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.cifar_bnn import CIFAR_BNN
from models.cifar_dfa import CIFARDFA
from utils.CIFAR10.data import get_cifar10_loaders, get_device
from utils.MNIST.seed import set_seed

from experiments.CIFAR10_experiments.results_utils import (
    append_result,
    Timer,
    count_parameters,
)


def train_dfa_epoch(
    model: CIFAR_BNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    dfa: CIFARDFA,
    device: torch.device,
    clipping_strategy: str = "layer_wise",
    max_grad_norm: float = 0.25,
) -> tuple[float, float]:
    """Train one epoch using manually calculated DFA gradients."""

    if clipping_strategy not in {"layer_wise", "global", "none"}:
        raise ValueError("Invalid clipping strategy.")

    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        intermediates = model.forward_with_intermediates(images)
        outputs = intermediates[0]

        loss = F.cross_entropy(outputs, labels)

        local_gradients = dfa.compute_local_gradients(
            model=model,
            x_input=images,
            labels=labels,
            forward_intermediates=intermediates,
        )

        # Print unclipped gradient diagnostics once per process.
        if not getattr(train_dfa_epoch, "_diagnostics_printed", False):
            print("\n--- DFA Gradient Diagnostics (Before Clipping) ---")

            for name, grad in local_gradients.items():
                grad_float = grad.detach().float()

                print(
                    f"{name:15s} | "
                    f"norm={grad_float.norm().item():.6e} | "
                    f"mean_abs={grad_float.abs().mean().item():.6e} | "
                    f"max_abs={grad_float.abs().max().item():.6e}"
                )

            combined_norm = torch.sqrt(
                sum(
                    grad.detach().float().norm().square()
                    for grad in local_gradients.values()
                )
            )

            print(f"Combined gradient norm: {combined_norm.item():.6e}")
            print("-----------------------------------------------\n")

            train_dfa_epoch._diagnostics_printed = True

        dfa.apply_gradients(model, local_gradients)

        if clipping_strategy == "layer_wise":
            for layer in (
                model.conv1,
                model.conv2,
                model.fc1,
                model.fc2,
            ):
                torch.nn.utils.clip_grad_norm_(
                    layer.parameters(),
                    max_norm=max_grad_norm,
                )

        elif clipping_strategy == "global":
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )

        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError("Training loader is empty.")

    return (
        total_loss / total,
        100.0 * correct / total,
    )


@torch.no_grad()
def evaluate_backbone(
    model: CIFAR_BNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate classification loss and accuracy."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError("Evaluation loader is empty.")

    return (
        total_loss / total,
        100.0 * correct / total,
    )


@torch.no_grad()
def collect_hidden_features(
    model: CIFAR_BNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract deterministic 128-dimensional representations and labels."""

    model.eval()

    features = []
    labels_all = []

    for images, labels in loader:
        images = images.to(device)

        hidden = model.extract_features(images)

        features.append(hidden.detach().cpu().float())
        labels_all.append(labels.detach().cpu().long())

    if not features:
        raise ValueError("Feature extraction loader is empty.")

    return (
        torch.cat(features, dim=0).contiguous(),
        torch.cat(labels_all, dim=0).contiguous(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 BNN + DFA training"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--clipping",
        choices=["layer_wise", "global", "none"],
        default="layer_wise",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.25)

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    if args.lr <= 0 or args.max_grad_norm <= 0:
        parser.error("Learning rate and max gradient norm must be positive.")

    set_seed(args.seed)

    # Reset diagnostics when main is invoked again in the same process.
    train_dfa_epoch._diagnostics_printed = False

    device = get_device()

    print(f"Using seed: {args.seed}")
    print(f"Using device: {device}")
    print(f"Clipping strategy: {args.clipping}")

    loaders = get_cifar10_loaders(
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
    )

    train_loader = loaders["train_loader"]
    val_loader = loaders["validation_loader"]
    test_loader = loaders["test_loader"]

    model = CIFAR_BNN().to(device)
    dfa = CIFARDFA(device=device, seed=args.seed)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    print("\nTraining CIFAR-10 BNN + DFA...\n")

    with Timer() as timer:
        for epoch in range(args.epochs):
            train_loss, train_acc = train_dfa_epoch(
                model,
                train_loader,
                optimizer,
                dfa,
                device,
                clipping_strategy=args.clipping,
                max_grad_norm=args.max_grad_norm,
            )

            val_loss, val_acc = evaluate_backbone(
                model,
                val_loader,
                device,
            )

            print(
                f"Epoch {epoch + 1:02d}/{args.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} "
                f"Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} "
                f"Acc: {val_acc:.2f}%"
            )

    test_loss, test_acc = evaluate_backbone(
        model,
        test_loader,
        device,
    )

    print(
        f"\nFinal Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_acc:.2f}%\n"
    )

    param_count = count_parameters(model)

    append_result(
        experiment="cifar_bnn_dfa",
        method="BNN + DFA",
        seed=args.seed,
        test_accuracy=test_acc,
        test_loss=test_loss,
        training_time_sec=timer.seconds,
        parameter_count=param_count,
        backbone_training_time_sec=timer.seconds,
        feature_extraction_time_sec=0.0,
        head_fit_time_sec=0.0,
        current_run_total_time_sec=timer.seconds,
        end_to_end_training_time_sec=timer.seconds,
        pipeline_timing_status="complete",
        backbone_parameter_count=param_count,
        head_parameter_count=0,
        total_parameter_count=param_count,
        n_train=len(loaders["split_indices"]["train"]),
        n_validation=len(loaders["split_indices"]["validation"]),
        n_test=len(loaders["split_indices"]["test"]),
    )


if __name__ == "__main__":
    main()