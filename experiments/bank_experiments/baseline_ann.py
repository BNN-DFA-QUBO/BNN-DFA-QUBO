"""Experiment 1: Baseline ANN + Backpropagation on Bank Marketing Dataset."""
from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from models.bank_bnn import BankANN
from utils.bank.data import get_bank_loaders, NUM_FEATURES, NUM_CLASSES
from utils.bank.metrics import compute_classification_metrics
from utils.bank.seed import set_seed, get_seed
from experiments.bank_experiments.results_utils import (
    append_bank_result,
    Timer,
    count_parameters,
)


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader.dataset)
    metrics = compute_classification_metrics(all_labels, all_preds)
    return avg_loss, metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_labels, all_preds, all_scores)
    return avg_loss, metrics


def run_baseline_ann(
    seed: int = 42,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    hidden_size: int = 64,
    device_str: str | None = None,
):
    if device_str is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device_str)

    set_seed(seed)
    print(f"[Experiment 1: Baseline ANN] Running with seed {seed} on {device}")

    train_loader, val_loader, test_loader = get_bank_loaders(batch_size=batch_size)

    model = BankANN(input_size=NUM_FEATURES, hidden_size=hidden_size, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    with Timer() as training_timer:
        for epoch in range(epochs):
            train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.2f}% | Train BA: {train_metrics['balanced_accuracy']:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | Val BA: {val_metrics['balanced_accuracy']:.2f}%"
            )

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"\nFinal Test Results: "
        f"Acc: {test_metrics['accuracy']:.2f}% | BA: {test_metrics['balanced_accuracy']:.2f}% | "
        f"F1: {test_metrics['f1']:.2f}% | ROC-AUC: {test_metrics.get('roc_auc', 0):.2f}% | PR-AUC: {test_metrics.get('pr_auc', 0):.2f}%"
    )

    n_params = count_parameters(model)
    append_bank_result(
        experiment="baseline_ann",
        method="Standard ANN + Backpropagation",
        seed=seed,
        test_accuracy=test_metrics["accuracy"],
        balanced_accuracy=test_metrics["balanced_accuracy"],
        f1_score=test_metrics["f1"],
        precision=test_metrics["precision"],
        recall=test_metrics["recall"],
        roc_auc=test_metrics.get("roc_auc"),
        pr_auc=test_metrics.get("pr_auc"),
        test_loss=test_loss,
        training_time_sec=training_timer.seconds,
        parameter_count=n_params,
        backbone_training_time_sec=training_timer.seconds,
        end_to_end_training_time_sec=training_timer.seconds,
        current_run_total_time_sec=training_timer.seconds,
        total_parameter_count=n_params,
        notes=f"epochs={epochs}, lr={lr}, hidden={hidden_size}",
    )
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: Baseline ANN + BP for Bank Dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    args = parser.parse_args()

    run_baseline_ann(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
    )

