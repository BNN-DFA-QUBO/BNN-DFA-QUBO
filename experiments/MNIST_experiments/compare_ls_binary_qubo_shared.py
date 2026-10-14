"""Controlled head comparison on one trained BNN+DFA representation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.MNIST_experiments.bnn_dfa_qubo import (
    calculate_final_scale_and_bias,
    collect_hidden_features,
    create_binary_targets,
    train_dfa,
)
from models.bnn import BNN
from models.dfa import DFAClassifier
from models.ls_heads import (
    accuracy_from_scores,
    fit_ridge_classifier,
    predict_scores,
)
from models.qubo_head import (
    calculate_ridge_ls_scales,
    optimize_binary_classifier,
)
from utils.MNIST.data import get_mnist_loaders
from utils.MNIST.seed import set_seed


def fit_binarized_ridge(H_train, y_train, ridge_weights, num_classes=10):
    """Apply the corrected Ridge LS binarization and training-only refit."""
    classifier = ridge_weights.detach().to(device="cpu", dtype=torch.float32)
    binary_weights = classifier[:-1, :].sign()
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

        # Match dfa_binary_head.py: keep scales non-negative by flipping the
        # binary direction, while computing the intercept from the raw scale.
        if scale_raw.item() < 0:
            binary_weights[:, class_id] *= -1
            scale = -scale_raw
        else:
            scale = scale_raw
        scales.append(scale)
        biases.append(target_mean - scale_raw * z_mean)

    return binary_weights, torch.stack(scales), torch.stack(biases)


def binary_scores(H, weights, scales, biases):
    return H.float() @ (weights * scales.unsqueeze(0)) + biases.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=161803)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--num-reads", type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1 or args.num_reads < 1:
        parser.error("--epochs and --num-reads must be positive")

    seed = set_seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using seed: {seed}\nUsing device: {device}")
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    model = BNN().to(device)
    dfa = DFAClassifier(hidden_size=128, output_size=10, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(args.epochs):
        loss, accuracy = train_dfa(model, train_loader, optimizer, dfa, device)
        print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {loss:.4f} | Train Accuracy: {accuracy:.2f}%")

    # Freeze model state and collect the representations once for each split.
    model.eval()
    H_train, y_train = collect_hidden_features(model, train_loader, device)
    H_test, y_test = collect_hidden_features(model, test_loader, device)
    print(f"Frozen representations: train={tuple(H_train.shape)}, test={tuple(H_test.shape)}")

    ridge = fit_ridge_classifier(H_train, y_train, num_classes=10, lambda_relative=1e-4)
    results = []

    def record(name, train_scores, test_scores, config):
        results.append({
            "seed": seed,
            "epochs": args.epochs,
            "num_reads": args.num_reads,
            "head": name,
            "train_accuracy": accuracy_from_scores(train_scores, y_train),
            "test_accuracy": accuracy_from_scores(test_scores, y_test),
            "configuration": config,
        })

    record("Real Ridge LS", predict_scores(H_train, ridge), predict_scores(H_test, ridge),
           "lambda_relative=1e-4; unpenalized intercept")

    bin_weights, bin_scales, bin_bias = fit_binarized_ridge(H_train, y_train, ridge)
    record("Binarized Ridge LS (refit scale+bias)",
           binary_scores(H_train, bin_weights, bin_scales, bin_bias),
           binary_scores(H_test, bin_weights, bin_scales, bin_bias),
           "sign(Ridge feature weights); one-vs-rest {0,1} training refit; negative scales flip binary weights")

    qubo_scales = calculate_ridge_ls_scales(
        H_train, y_train, num_classes=10, lambda_relative=1e-4
    )
    signed_targets = create_binary_targets(y_train, num_classes=10)
    qubo_weights = optimize_binary_classifier(
        H_train, signed_targets, qubo_scales, num_classes=10,
        num_reads=args.num_reads, seed=seed,
    )
    qubo_weights, qubo_final_scales, qubo_biases = calculate_final_scale_and_bias(
        H_train, y_train, qubo_weights, num_classes=10
    )
    record("QUBO Binary (Ridge LS scale init)",
           binary_scores(H_train, qubo_weights, qubo_final_scales, qubo_biases),
           binary_scores(H_test, qubo_weights, qubo_final_scales, qubo_biases),
           f"ridge lambda_relative=1e-4 scale init; signed {{-1,+1}} targets; simulated annealing num_reads={args.num_reads}; seed={seed}")

    baseline = results[0]["test_accuracy"]
    print("\nHead                                      Train Acc.   Test Acc.   Δ Test vs Ridge (pp)")
    print("-" * 91)
    for row in results:
        delta = row["test_accuracy"] - baseline
        print(f"{row['head']:<42} {row['train_accuracy']:>8.2f}%   {row['test_accuracy']:>8.2f}%   {delta:>+9.2f}")

    output_dir = Path("data/results_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"compare_ls_binary_qubo_shared_{timestamp}.csv"
    columns = ["seed", "epochs", "num_reads", "head", "train_accuracy", "test_accuracy", "test_accuracy_diff_vs_ridge_pp", "configuration"]
    with output_path.open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({**row, "test_accuracy_diff_vs_ridge_pp": row["test_accuracy"] - baseline})
    print(f"\nDetailed results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
