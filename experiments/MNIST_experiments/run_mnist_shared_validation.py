"""Shared-representation MNIST head comparison with validation-only STE checkpointing.

This diagnostic runner trains one BNN+DFA backbone per seed, creates a stratified
validation split from the official MNIST training partition, then compares all
heads on the same frozen features. The official MNIST test partition is only
used for final reporting; it never selects a checkpoint or threshold.

Example:
    python -m experiments.MNIST_experiments.run_mnist_shared_validation --seed 123 --epochs 16 --num-reads 100
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from experiments.MNIST_experiments.bnn_dfa_qubo import (
    calculate_final_scale_and_bias,
    collect_hidden_features,
    create_binary_targets,
    train_dfa,
)
from experiments.MNIST_experiments.compare_ls_binary_qubo_shared import (
    binary_scores,
    fit_binarized_ridge,
)
from models.bnn import BNN
from models.dfa import DFAClassifier
from models.ls_heads import accuracy_from_scores, fit_ridge_classifier, predict_scores
from models.qubo_head import calculate_ridge_ls_scales, optimize_binary_classifier
from utils.MNIST.data import get_mnist_loaders
from utils.MNIST.seed import set_seed
from experiments.MNIST_experiments.dfa_binary_head_ste import binary_ste


def _collect_ordered(model, loader, device):
    """Collect features in loader order; all head labels stay aligned."""
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            x = torch.flatten(images, start_dim=1)
            features.append(torch.relu(model.fc1(x)).cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(features), torch.cat(labels)


def _train_ste_with_validation(H_train, y_train, H_val, y_val, epochs=50, lr=0.01):
    """Fit STE weights and retain the epoch with highest validation accuracy."""
    H_train, H_val = H_train.float(), H_val.float()
    y_train, y_val = y_train.long(), y_val.long()
    hidden_size = H_train.shape[1]
    num_classes = int(torch.max(y_train).item()) + 1

    latent = (torch.randn(hidden_size, num_classes) * 0.01).requires_grad_()
    bias = torch.zeros(num_classes, requires_grad=True)
    optimizer = torch.optim.Adam([latent, bias], lr=lr)
    best = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = H_train @ binary_ste(latent) + bias
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_logits = H_val @ binary_ste(latent) + bias
            val_accuracy = (val_logits.argmax(1) == y_val).float().mean().item()
            # Stable tie-break: keep the earliest epoch.
            if best is None or val_accuracy > best["val_accuracy"]:
                best = {
                    "val_accuracy": val_accuracy,
                    "epoch": epoch + 1,
                    "weights": torch.where(latent >= 0, torch.ones_like(latent), -torch.ones_like(latent)).detach().clone(),
                    "bias": bias.detach().clone(),
                }

    return best


def _append_result(rows, name, H_train, y_train, H_val, y_val, H_test, y_test,
                   train_scores, val_scores, test_scores, config):
    rows.append({
        "head": name,
        "train_accuracy_pct": accuracy_from_scores(train_scores, y_train),
        "validation_accuracy_pct": accuracy_from_scores(val_scores, y_val),
        "test_accuracy_pct": accuracy_from_scores(test_scores, y_test),
        "configuration": config,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=16, help="BNN+DFA backbone epochs")
    parser.add_argument("--ste-epochs", type=int, default=50)
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if args.epochs < 1 or args.ste_epochs < 1 or args.num_reads < 1:
        parser.error("epochs, ste-epochs, and num-reads must be positive")
    if not 0.05 <= args.validation_fraction <= 0.4:
        parser.error("validation-fraction must be between 0.05 and 0.4")

    seed = set_seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Seed={seed} | device={device}")
    base_train_loader, test_loader = get_mnist_loaders(batch_size=64)
    dataset = base_train_loader.dataset
    targets = np.asarray(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=args.validation_fraction,
        random_state=seed,
        stratify=targets,
    )
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()), batch_size=64, shuffle=True,
        generator=train_generator, num_workers=0,
    )
    train_eval_loader = DataLoader(
        Subset(dataset, train_indices.tolist()), batch_size=256, shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()), batch_size=256, shuffle=False,
        num_workers=0,
    )

    model = BNN().to(device)
    dfa = DFAClassifier(hidden_size=128, output_size=10, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(args.epochs):
        loss, accuracy = train_dfa(model, train_loader, optimizer, dfa, device)
        print(f"Backbone epoch {epoch + 1}/{args.epochs}: loss={loss:.4f}, train_acc={accuracy:.2f}%")

    H_train, y_train = _collect_ordered(model, train_eval_loader, device)
    H_val, y_val = _collect_ordered(model, val_loader, device)
    H_test, y_test = _collect_ordered(model, test_loader, device)
    print(f"Frozen features: train={tuple(H_train.shape)}, val={tuple(H_val.shape)}, test={tuple(H_test.shape)}")

    rows = []
    ridge = fit_ridge_classifier(H_train, y_train, num_classes=10, lambda_relative=1e-4)
    ridge_train, ridge_val, ridge_test = (
        predict_scores(H_train, ridge), predict_scores(H_val, ridge), predict_scores(H_test, ridge)
    )
    _append_result(rows, "Ridge LS", H_train, y_train, H_val, y_val, H_test, y_test,
                   ridge_train, ridge_val, ridge_test, "lambda_relative=1e-4; unpenalized intercept")

    bin_w, bin_scale, bin_bias = fit_binarized_ridge(H_train, y_train, ridge)
    bin_train = binary_scores(H_train, bin_w, bin_scale, bin_bias)
    bin_val = binary_scores(H_val, bin_w, bin_scale, bin_bias)
    bin_test = binary_scores(H_test, bin_w, bin_scale, bin_bias)
    _append_result(rows, "Binarized Ridge LS", H_train, y_train, H_val, y_val, H_test, y_test,
                   bin_train, bin_val, bin_test, "binary signs with training-only scale/bias refit")

    signed_targets = create_binary_targets(y_train, num_classes=10)
    qubo_scales = calculate_ridge_ls_scales(H_train, y_train, num_classes=10, lambda_relative=1e-4)
    # Avoid a zero alpha making a class objective constant; log the effective floor.
    qubo_scales = torch.where(qubo_scales > 1e-8, qubo_scales, torch.full_like(qubo_scales, 1e-8))
    qubo_w = optimize_binary_classifier(
        H_train, signed_targets, qubo_scales, num_classes=10,
        num_reads=args.num_reads, seed=seed,
    )
    qubo_w, qubo_scale, qubo_bias = calculate_final_scale_and_bias(H_train, y_train, qubo_w, num_classes=10)
    qubo_train = H_train.float() @ (qubo_w * qubo_scale.unsqueeze(0)) + qubo_bias
    qubo_val = H_val.float() @ (qubo_w * qubo_scale.unsqueeze(0)) + qubo_bias
    qubo_test = H_test.float() @ (qubo_w * qubo_scale.unsqueeze(0)) + qubo_bias
    _append_result(rows, "QUBO Binary", H_train, y_train, H_val, y_val, H_test, y_test,
                   qubo_train, qubo_val, qubo_test, f"Ridge scale init; signed targets; num_reads={args.num_reads}")

    ste = _train_ste_with_validation(H_train, y_train, H_val, y_val, epochs=args.ste_epochs)
    ste_train = H_train.float() @ ste["weights"] + ste["bias"]
    ste_val = H_val.float() @ ste["weights"] + ste["bias"]
    ste_test = H_test.float() @ ste["weights"] + ste["bias"]
    _append_result(rows, "STE (best validation epoch)", H_train, y_train, H_val, y_val, H_test, y_test,
                   ste_train, ste_val, ste_test, f"epochs={args.ste_epochs}; selected_epoch={ste['epoch']}")

    output_dir = Path("data/results_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"shared_validation_heads_seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields = ["seed", "backbone_epochs", "validation_fraction", "num_reads", "ste_epochs",
              "head", "train_accuracy_pct", "validation_accuracy_pct", "test_accuracy_pct", "configuration"]
    with output_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "seed": seed, "backbone_epochs": args.epochs,
                "validation_fraction": args.validation_fraction, "num_reads": args.num_reads,
                "ste_epochs": args.ste_epochs, **row,
            })

    print("\nHead comparison (same frozen backbone/features):")
    print(f"{'Head':32s} {'Train %':>9s} {'Validation %':>14s} {'Test %':>9s}")
    for row in rows:
        print(f"{row['head']:32s} {row['train_accuracy_pct']:9.2f} {row['validation_accuracy_pct']:14.2f} {row['test_accuracy_pct']:9.2f}")
    print(f"Selected STE epoch by validation accuracy: {ste['epoch']}")
    print(f"CSV: {output_path.resolve()}")


if __name__ == "__main__":
    main()
