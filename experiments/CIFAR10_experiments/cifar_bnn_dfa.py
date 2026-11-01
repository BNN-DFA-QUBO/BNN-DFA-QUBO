"""Train CIFAR-10 BNN backbone with Direct Feedback Alignment (DFA).

Standalone diagnostic runner and reusable training/feature-collection routines.
"""
from __future__ import annotations

import argparse
import time
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
) -> tuple[float, float]:
    """Train one epoch of CIFAR_BNN using layer-wise DFA updates."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass returning all pre- and post-activation tensors
        intermediates = model.forward_with_intermediates(images)
        outputs = intermediates[0]

        loss = F.cross_entropy(outputs, labels)

        # Compute layer-wise DFA feedback gradients
        local_gradients = dfa.compute_local_gradients(
            model=model,
            x_input=images,
            labels=labels,
            forward_intermediates=intermediates,
        )

        # Assign local gradients to model parameters
        dfa.apply_gradients(model, local_gradients)

        # Update latent weights
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_backbone(
    model: CIFAR_BNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate CIFAR_BNN classification accuracy and cross-entropy loss."""
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

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def collect_hidden_features(
    model: CIFAR_BNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract deterministic 128-dimensional representations and aligned labels."""
    model.eval()
    features = []
    labels_all = []

    for images, labels in loader:
        images = images.to(device)
        h = model.extract_features(images)
        features.append(h.cpu())
        labels_all.append(labels.cpu())

    return torch.cat(features, dim=0), torch.cat(labels_all, dim=0)


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 BNN + DFA training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using seed: {args.seed}")
    print(f"Using device: {device}")

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("\nTraining CIFAR-10 BNN + DFA...\n")
    with Timer() as timer:
        for epoch in range(args.epochs):
            train_loss, train_acc = train_dfa_epoch(
                model, train_loader, optimizer, dfa, device
            )
            val_loss, val_acc = evaluate_backbone(model, val_loader, device)
            print(
                f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
            )

    test_loss, test_acc = evaluate_backbone(model, test_loader, device)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%\n")

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
