"""Sweep QUBO read counts over shared frozen BNN+DFA representations."""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

from experiments.MNIST_experiments.bnn_dfa_qubo import (
    calculate_final_scale_and_bias,
    collect_hidden_features,
    create_binary_targets,
    train_dfa,
)
from models.bnn import BNN
from models.dfa import DFAClassifier
from models.ls_heads import accuracy_from_scores
from models.qubo_head import calculate_ridge_ls_scales, optimize_binary_classifier
from utils.MNIST.data import get_mnist_loaders
from utils.MNIST.seed import set_seed


def head_scores(H, binary_weights, scales, biases):
    return H.float() @ (binary_weights.float() * scales.float().unsqueeze(0)) + biases.float().unsqueeze(0)


def write_csv_exclusive(path: Path, columns: list[str], rows: list[dict]):
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--reads", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_sweeps < 1 or any(read < 1 for read in args.reads):
        parser.error("--epochs, --batch-size, --num-sweeps, and all --reads values must be positive")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    detailed_rows = []
    lambda_relative = 1e-4

    for seed in args.seeds:
        set_seed(seed)
        print(f"\n=== Seed {seed}: training one BNN+DFA backbone ===")
        train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
        model = BNN().to(device)
        dfa = DFAClassifier(hidden_size=128, output_size=10, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(args.epochs):
            loss, accuracy = train_dfa(model, train_loader, optimizer, dfa, device)
            print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {loss:.4f} | Train Accuracy: {accuracy:.2f}%")

        model.eval()
        H_train, y_train = collect_hidden_features(model, train_loader, device)
        H_test, y_test = collect_hidden_features(model, test_loader, device)
        print(f"Frozen representations: train={tuple(H_train.shape)}, test={tuple(H_test.shape)}")

        # One fixed Ridge-LS scale initialization and signed target matrix per seed.
        scales = calculate_ridge_ls_scales(
            H_train, y_train, num_classes=10, lambda_relative=lambda_relative
        )
        signed_targets = create_binary_targets(y_train, num_classes=10)

        for read_count in args.reads:
            print(f"Seed {seed}: optimizing with num_reads={read_count}")
            start = time.perf_counter()
            binary_weights = optimize_binary_classifier(
                H_train,
                signed_targets,
                scales,
                num_classes=10,
                num_reads=read_count,
                seed=seed,
                num_sweeps=args.num_sweeps,
            )
            optimization_seconds = time.perf_counter() - start

            # Scale and bias refitting is measured separately from annealing.
            binary_weights, final_scales, biases = calculate_final_scale_and_bias(
                H_train, y_train, binary_weights, num_classes=10
            )
            train_accuracy = accuracy_from_scores(
                head_scores(H_train, binary_weights, final_scales, biases), y_train
            )
            test_accuracy = accuracy_from_scores(
                head_scores(H_test, binary_weights, final_scales, biases), y_test
            )
            detailed_rows.append({
                "seed": seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_reads": read_count,
                "num_sweeps": args.num_sweeps,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "qubo_optimization_seconds": optimization_seconds,
                "configuration": (
                    "BNN+DFA hidden_size=128 output_size=10 Adam lr=0.001; "
                    f"Ridge-LS scale lambda_relative={lambda_relative}; "
                    f"signed {{-1,+1}} targets; num_sweeps={args.num_sweeps}"
                ),
            })
            print(
                f"Seed {seed} | reads={read_count} | sweeps={args.num_sweeps} | train={train_accuracy:.2f}% | "
                f"test={test_accuracy:.2f}% | QUBO={optimization_seconds:.3f}s"
            )

    grouped = defaultdict(list)
    for row in detailed_rows:
        grouped[row["num_reads"]].append(row)
    aggregate_rows = []
    for read_count in args.reads:
        group = grouped[read_count]
        accuracies = [row["test_accuracy"] for row in group]
        runtimes = [row["qubo_optimization_seconds"] for row in group]
        aggregate_rows.append({
            "num_reads": read_count,
            "seed_count": len(group),
            "mean_test_accuracy": statistics.mean(accuracies),
            "sample_std_test_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else "",
            "mean_qubo_optimization_seconds": statistics.mean(runtimes),
            "sample_std_qubo_optimization_seconds": statistics.stdev(runtimes) if len(runtimes) > 1 else "",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seeds": " ".join(map(str, args.seeds)),
            "configuration": f"Ridge-LS scale lambda_relative={lambda_relative}; BNN+DFA lr=0.001; num_sweeps={args.num_sweeps}",
        })

    output_dir = Path("data/results_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    detailed_path = output_dir / f"sweep_qubo_num_reads_shared_{timestamp}.csv"
    aggregate_path = output_dir / f"sweep_qubo_num_reads_shared_aggregate_{timestamp}.csv"
    write_csv_exclusive(
        detailed_path,
        ["seed", "epochs", "batch_size", "num_reads", "num_sweeps", "train_accuracy", "test_accuracy", "qubo_optimization_seconds", "configuration"],
        detailed_rows,
    )
    write_csv_exclusive(
        aggregate_path,
        ["num_reads", "num_sweeps", "seed_count", "mean_test_accuracy", "sample_std_test_accuracy", "mean_qubo_optimization_seconds", "sample_std_qubo_optimization_seconds", "epochs", "batch_size", "seeds", "configuration"],
        aggregate_rows,
    )
    print(f"\nDetailed results saved to: {detailed_path.resolve()}")
    print(f"Aggregate results saved to: {aggregate_path.resolve()}")


if __name__ == "__main__":
    main()
