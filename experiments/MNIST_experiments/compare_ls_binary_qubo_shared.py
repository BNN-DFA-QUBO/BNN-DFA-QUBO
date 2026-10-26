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
    create_binary_targets,
)
from experiments.MNIST_experiments.mnist_representation import load_or_train_mnist_representation
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
from utils.MNIST.representation import DEFAULT_REPRESENTATION_DIR
from experiments.MNIST_experiments.tune_qubo_validation import (
    DEFAULT_NUM_READS, DEFAULT_NUM_SWEEPS, DEFAULT_SCALE_MULTIPLIERS, _parse_grid,
    tune_from_representations, validate_grid, write_tuning_csv,
)


def fit_binarized_ridge(H_train, y_train, ridge_weights, num_classes=10):
    """Bipolarize Ridge features (zero maps to +1) and refit on training rows."""
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
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--representation-dir", type=Path, default=DEFAULT_REPRESENTATION_DIR)
    parser.add_argument("--fixed-qubo", action="store_true")
    parser.add_argument("--num-reads-grid", type=lambda s: _parse_grid(s, int, "num-reads-grid"),
                        default=DEFAULT_NUM_READS)
    parser.add_argument("--num-sweeps-grid", type=lambda s: _parse_grid(s, int, "num-sweeps-grid"),
                        default=DEFAULT_NUM_SWEEPS)
    parser.add_argument("--scale-multipliers", type=lambda s: _parse_grid(s, float, "scale-multipliers"),
                        default=DEFAULT_SCALE_MULTIPLIERS)
    args = parser.parse_args()
    if args.epochs < 1 or args.num_reads < 1 or args.num_sweeps < 1:
        parser.error("--epochs, --num-reads, and --num-sweeps must be positive")
    if not 0.05 <= args.validation_fraction <= 0.4:
        parser.error("--validation-fraction must be between 0.05 and 0.4")
    if not args.fixed_qubo:
        try:
            validate_grid(args.num_reads_grid, args.num_sweeps_grid, args.scale_multipliers)
        except ValueError as exc:
            parser.error(str(exc))

    seed = set_seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using seed: {seed}\nUsing device: {device}")
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    representation, representation_path, representation_source = load_or_train_mnist_representation(
        train_loader.dataset, test_loader.dataset, seed, args.epochs,
        args.validation_fraction, device, args.representation_dir,
    )
    H_train, y_train = representation["H_train"], representation["y_train"]
    H_val, y_val = representation["H_validation"], representation["y_validation"]
    H_test, y_test = representation["H_test"], representation["y_test"]
    metadata = representation["metadata"]
    print(f"Frozen representations: train={tuple(H_train.shape)}, val={tuple(H_val.shape)}, test={tuple(H_test.shape)}")
    print(f"Representation: {representation_path.resolve()} ({representation_source}, id={representation['artifact_id']})")

    ridge = fit_ridge_classifier(H_train, y_train, num_classes=10, lambda_relative=1e-4)
    results = []

    def record(name, train_scores, val_scores, test_scores, config):
        results.append({
            "seed": seed,
            "epochs": args.epochs,
            "num_reads": args.num_reads,
            "head": name,
            "train_accuracy": accuracy_from_scores(train_scores, y_train),
            "validation_accuracy": accuracy_from_scores(val_scores, y_val),
            "test_accuracy": accuracy_from_scores(test_scores, y_test),
            "configuration": config,
            "representation_artifact_id": representation["artifact_id"],
            "representation_path": str(representation_path.resolve()),
            "representation_cache_status": representation_source,
            "n_train": metadata["sample_counts"]["train"],
            "n_validation": metadata["sample_counts"]["validation"],
            "n_test": metadata["sample_counts"]["test"],
            "split_protocol": metadata["split_protocol"],
        })

    record("Real Ridge LS", predict_scores(H_train, ridge), predict_scores(H_val, ridge), predict_scores(H_test, ridge),
           "lambda_relative=1e-4; unpenalized intercept")

    bin_weights, bin_scales, bin_bias = fit_binarized_ridge(H_train, y_train, ridge)
    record("Binarized Ridge LS (refit scale+bias)",
           binary_scores(H_train, bin_weights, bin_scales, bin_bias),
           binary_scores(H_val, bin_weights, bin_scales, bin_bias),
           binary_scores(H_test, bin_weights, bin_scales, bin_bias),
           "sign(Ridge feature weights); one-vs-rest {0,1} training refit; negative scales flip binary weights")

    tuning_rows = None
    tuning_winner = None
    if args.fixed_qubo:
        qubo_scales = calculate_ridge_ls_scales(H_train, y_train, num_classes=10, lambda_relative=1e-4)
        qubo_scales = torch.where(qubo_scales > 1e-8, qubo_scales, torch.full_like(qubo_scales, 1e-8))
        signed_targets = create_binary_targets(y_train, num_classes=10)
        qubo_weights = optimize_binary_classifier(H_train, signed_targets, qubo_scales,
                                                  num_classes=10, num_reads=args.num_reads,
                                                  num_sweeps=args.num_sweeps, seed=seed)
        qubo_weights, qubo_final_scales, qubo_biases = calculate_final_scale_and_bias(
            H_train, y_train, qubo_weights, num_classes=10)
        qubo_config = f"fixed num_reads={args.num_reads}; num_sweeps={args.num_sweeps}; seed={seed}"
    else:
        tuning_winner, tuning_rows = tune_from_representations(
            H_train, y_train, H_val, y_val, seed,
            args.num_reads_grid, args.num_sweeps_grid, args.scale_multipliers,
        )
        qubo_weights = tuning_winner["_weights"]
        qubo_final_scales = tuning_winner["_scales"]
        qubo_biases = tuning_winner["_bias"]
        qubo_config = (f"selected num_reads={tuning_winner['num_reads']}; "
                       f"num_sweeps={tuning_winner['num_sweeps']}; "
                       f"scale_multiplier={tuning_winner['scale_multiplier']}; "
                       f"solver_seed={tuning_winner['solver_seed']}; "
                       f"validation_accuracy={tuning_winner['validation_accuracy']:.4f}%")
    qubo_row_start = len(results)
    record("QUBO Binary (Ridge LS scale init)",
           binary_scores(H_train, qubo_weights, qubo_final_scales, qubo_biases),
           binary_scores(H_val, qubo_weights, qubo_final_scales, qubo_biases),
           binary_scores(H_test, qubo_weights, qubo_final_scales, qubo_biases),
           f"ridge lambda_relative=1e-4 scale init; signed {{-1,+1}} targets; {qubo_config}")
    if tuning_winner:
        results[qubo_row_start].update({
            "qubo_num_reads_selected": tuning_winner["num_reads"],
            "qubo_num_sweeps_selected": tuning_winner["num_sweeps"],
            "qubo_grid_protocol": "fixed_27_configuration_v1",
            "qubo_scale_multiplier_selected": tuning_winner["scale_multiplier"],
            "qubo_validation_accuracy": tuning_winner["validation_accuracy"],
            "qubo_solver_seed_selected": tuning_winner["solver_seed"],
            "qubo_selection_metric": "validation_accuracy_percent",
        })

    baseline = results[0]["test_accuracy"]
    print("\nHead                                      Train Acc.   Validation   Test Acc.   Δ Test vs Ridge (pp)")
    print("-" * 91)
    for row in results:
        delta = row["test_accuracy"] - baseline
        print(f"{row['head']:<42} {row['train_accuracy']:>8.2f}%   {row['validation_accuracy']:>8.2f}%   {row['test_accuracy']:>8.2f}%   {delta:>+9.2f}")

    output_dir = Path("data/results_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    protocol_name = "27config" if not args.fixed_qubo else "fixed_qubo"
    output_path = output_dir / f"compare_ls_binary_qubo_shared_{protocol_name}_{timestamp}.csv"
    columns = ["seed", "epochs", "validation_fraction", "num_reads", "num_sweeps", "head", "train_accuracy",
               "validation_accuracy", "test_accuracy", "test_accuracy_diff_vs_ridge_pp", "configuration",
               "representation_artifact_id", "representation_path", "representation_cache_status",
               "n_train", "n_validation", "n_test", "split_protocol", "qubo_num_reads_selected",
               "qubo_num_sweeps_selected",
               "qubo_scale_multiplier_selected", "qubo_validation_accuracy", "qubo_solver_seed_selected",
               "qubo_selection_metric"]
    columns.append("qubo_grid_protocol")
    with output_path.open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({"validation_fraction": args.validation_fraction,
                             "num_sweeps": args.num_sweeps, **row,
                             "test_accuracy_diff_vs_ridge_pp": row["test_accuracy"] - baseline})
    print(f"\nDetailed results saved to: {output_path.resolve()}")
    if tuning_rows is not None:
        tuning_path = write_tuning_csv(
            tuning_rows, seed, args.epochs, args.validation_fraction, output_dir,
            representation["artifact_id"], metadata["sample_counts"]["train"],
            metadata["sample_counts"]["validation"], metadata["sample_counts"]["test"],
        )
        print(f"QUBO tuning configurations saved to: {tuning_path.resolve()}")


if __name__ == "__main__":
    main()
