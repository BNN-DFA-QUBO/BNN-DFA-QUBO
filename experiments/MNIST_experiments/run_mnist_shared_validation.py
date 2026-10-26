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

import torch
import torch.nn.functional as F
from experiments.MNIST_experiments.bnn_dfa_qubo import (
    calculate_final_scale_and_bias,
    create_binary_targets,
)
from experiments.MNIST_experiments.compare_ls_binary_qubo_shared import (
    binary_scores,
    fit_binarized_ridge,
)
from models.ls_heads import accuracy_from_scores, fit_ridge_classifier, predict_scores
from models.qubo_head import calculate_ridge_ls_scales, optimize_binary_classifier
from utils.MNIST.data import get_mnist_loaders
from utils.MNIST.seed import set_seed
from experiments.MNIST_experiments.dfa_binary_head_ste import binary_ste
from experiments.MNIST_experiments.results_utils import append_result
from experiments.MNIST_experiments.mnist_representation import load_or_train_mnist_representation
from experiments.MNIST_experiments.tune_qubo_validation import (
    DEFAULT_NUM_READS, DEFAULT_NUM_SWEEPS, DEFAULT_SCALE_MULTIPLIERS, _parse_grid,
    tune_from_representations, write_tuning_csv,
)
from utils.MNIST.representation import DEFAULT_REPRESENTATION_DIR


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


def _persist_selected_qubo_result(*, seed, test_accuracy, winner, tuning_rows,
                                  validation_fraction, representation,
                                  representation_path, representation_source,
                                  tuning_csv_path):
    """Append exactly one selected QUBO result through the shared results writer."""
    metadata = representation["metadata"]
    counts = metadata["sample_counts"]
    append_result(
        experiment="mnist_shared_validation",
        method="QUBO Binary",
        seed=seed,
        test_accuracy=test_accuracy,
        parameter_count=metadata.get("backbone_parameter_count"),
        qubo_time_sec=sum(row.get("runtime_sec", 0.0) for row in tuning_rows),
        qubo_tuning_enabled=True,
        qubo_num_reads_selected=winner["num_reads"],
        qubo_num_sweeps_selected=winner["num_sweeps"],
        qubo_grid_protocol="fixed_27_configuration_v1",
        qubo_scale_multiplier_selected=winner["scale_multiplier"],
        qubo_validation_accuracy=winner["validation_accuracy"],
        qubo_validation_fraction=validation_fraction,
        qubo_grid_size=len(tuning_rows),
        qubo_solver_seed_selected=winner["solver_seed"],
        qubo_selection_metric="validation_accuracy_percent",
        representation_artifact_id=representation["artifact_id"],
        representation_path=str(representation_path.resolve()),
        representation_cache_status=representation_source,
        n_train=counts["train"],
        n_validation=counts["validation"],
        n_test=counts["test"],
        notes=f"tuning_csv={tuning_csv_path.resolve()}",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=16, help="BNN+DFA backbone epochs")
    parser.add_argument("--ste-epochs", type=int, default=50)
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--representation-dir", type=Path, default=DEFAULT_REPRESENTATION_DIR)
    parser.add_argument("--fixed-qubo", action="store_true",
                        help="use the historical fixed num-reads QUBO instead of validation tuning")
    parser.add_argument("--num-reads-grid", type=lambda s: _parse_grid(s, int, "num-reads-grid"),
                        default=DEFAULT_NUM_READS)
    parser.add_argument("--num-sweeps-grid", type=lambda s: _parse_grid(s, int, "num-sweeps-grid"),
                        default=DEFAULT_NUM_SWEEPS)
    parser.add_argument("--scale-multipliers", type=lambda s: _parse_grid(s, float, "scale-multipliers"),
                        default=DEFAULT_SCALE_MULTIPLIERS)
    args = parser.parse_args()
    if args.epochs < 1 or args.ste_epochs < 1 or args.num_reads < 1 or args.num_sweeps < 1:
        parser.error("epochs, ste-epochs, num-reads, and num-sweeps must be positive")
    if not 0.05 <= args.validation_fraction <= 0.4:
        parser.error("validation-fraction must be between 0.05 and 0.4")
    if not args.fixed_qubo:
        try:
            from experiments.MNIST_experiments.tune_qubo_validation import validate_grid
            validate_grid(args.num_reads_grid, args.num_sweeps_grid, args.scale_multipliers)
        except ValueError as exc:
            parser.error(str(exc))

    seed = set_seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Seed={seed} | device={device}")
    base_train_loader, test_loader = get_mnist_loaders(batch_size=64)
    representation, representation_path, representation_source = load_or_train_mnist_representation(
        base_train_loader.dataset, test_loader.dataset, seed, args.epochs,
        args.validation_fraction, device, args.representation_dir,
    )
    H_train, y_train = representation["H_train"], representation["y_train"]
    H_val, y_val = representation["H_validation"], representation["y_validation"]
    H_test, y_test = representation["H_test"], representation["y_test"]
    provenance = representation["metadata"]
    artifact_id = representation["artifact_id"]
    sample_counts = provenance["sample_counts"]
    print(f"Representation: {representation_path.resolve()} ({representation_source}, id={artifact_id})")
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

    tuning_rows = None
    tuning_winner = None
    if args.fixed_qubo:
        signed_targets = create_binary_targets(y_train, num_classes=10)
        qubo_scales = calculate_ridge_ls_scales(H_train, y_train, num_classes=10, lambda_relative=1e-4)
        qubo_scales = torch.where(qubo_scales > 1e-8, qubo_scales, torch.full_like(qubo_scales, 1e-8))
        qubo_w = optimize_binary_classifier(
            H_train, signed_targets, qubo_scales, num_classes=10,
            num_reads=args.num_reads, num_sweeps=args.num_sweeps, seed=seed,
        )
        qubo_w, qubo_scale, qubo_bias = calculate_final_scale_and_bias(H_train, y_train, qubo_w, num_classes=10)
        qubo_config = f"fixed num_reads={args.num_reads}; num_sweeps={args.num_sweeps}; seed={seed}"
    else:
        tuning_winner, tuning_rows = tune_from_representations(
            H_train, y_train, H_val, y_val, seed,
            args.num_reads_grid, args.num_sweeps_grid, args.scale_multipliers,
        )
        qubo_w, qubo_scale, qubo_bias = (
            tuning_winner["_weights"], tuning_winner["_scales"], tuning_winner["_bias"]
        )
        qubo_config = (f"selected num_reads={tuning_winner['num_reads']}; "
                       f"num_sweeps={tuning_winner['num_sweeps']}; "
                       f"scale_multiplier={tuning_winner['scale_multiplier']}; "
                       f"solver_seed={tuning_winner['solver_seed']}; "
                       f"validation_accuracy={tuning_winner['validation_accuracy']:.4f}%")
    qubo_train = H_train.float() @ (qubo_w * qubo_scale.unsqueeze(0)) + qubo_bias
    qubo_val = H_val.float() @ (qubo_w * qubo_scale.unsqueeze(0)) + qubo_bias
    qubo_test = H_test.float() @ (qubo_w * qubo_scale.unsqueeze(0)) + qubo_bias
    _append_result(rows, "QUBO Binary", H_train, y_train, H_val, y_val, H_test, y_test,
                   qubo_train, qubo_val, qubo_test,
                   f"Ridge scale init; signed targets; {qubo_config}")

    # Keep head initialization independent of whether the representation was
    # trained now or loaded from cache.
    set_seed(seed)
    ste = _train_ste_with_validation(H_train, y_train, H_val, y_val, epochs=args.ste_epochs)
    ste_train = H_train.float() @ ste["weights"] + ste["bias"]
    ste_val = H_val.float() @ ste["weights"] + ste["bias"]
    ste_test = H_test.float() @ ste["weights"] + ste["bias"]
    _append_result(rows, "STE (best validation epoch)", H_train, y_train, H_val, y_val, H_test, y_test,
                   ste_train, ste_val, ste_test, f"epochs={args.ste_epochs}; selected_epoch={ste['epoch']}")

    output_dir = Path("data/results_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_name = "27config" if not args.fixed_qubo else "fixed_qubo"
    output_path = output_dir / f"shared_validation_heads_{protocol_name}_seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields = ["seed", "backbone_epochs", "validation_fraction", "num_reads", "num_sweeps", "ste_epochs",
              "head", "train_accuracy_pct", "validation_accuracy_pct", "test_accuracy_pct", "configuration",
              "representation_artifact_id", "representation_path", "representation_cache_status",
              "n_train", "n_validation", "n_test", "split_protocol", "dataset_fingerprint",
              "qubo_num_reads_selected", "qubo_scale_multiplier_selected", "qubo_validation_accuracy",
              "qubo_num_sweeps_selected", "qubo_solver_seed_selected", "qubo_selection_metric"]
    fields.append("qubo_grid_protocol")
    for row in rows:
        row.update({
            "representation_artifact_id": artifact_id,
            "representation_path": str(representation_path.resolve()),
            "representation_cache_status": representation_source,
            "n_train": sample_counts["train"], "n_validation": sample_counts["validation"],
            "n_test": sample_counts["test"], "split_protocol": provenance["split_protocol"],
            "dataset_fingerprint": provenance["dataset_fingerprint"],
        })
    qubo_row = next(row for row in rows if row["head"] == "QUBO Binary")
    if tuning_winner:
        qubo_row.update({
            "qubo_num_reads_selected": tuning_winner["num_reads"],
            "qubo_num_sweeps_selected": tuning_winner["num_sweeps"],
            "qubo_grid_protocol": "fixed_27_configuration_v1",
            "qubo_scale_multiplier_selected": tuning_winner["scale_multiplier"],
            "qubo_validation_accuracy": tuning_winner["validation_accuracy"],
            "qubo_solver_seed_selected": tuning_winner["solver_seed"],
            "qubo_selection_metric": "validation_accuracy_percent",
        })
    with output_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "seed": seed, "backbone_epochs": args.epochs,
                "validation_fraction": args.validation_fraction, "num_reads": args.num_reads,
                "num_sweeps": args.num_sweeps,
                "ste_epochs": args.ste_epochs, **row,
            })

    if tuning_rows is not None:
        tuning_path = write_tuning_csv(
            tuning_rows, seed, args.epochs, args.validation_fraction,
            output_dir, artifact_id, sample_counts["train"],
            sample_counts["validation"], sample_counts["test"],
        )
        print(f"QUBO tuning CSV: {tuning_path.resolve()}")
        # The shared-validation runner owns comparison artifacts; the shared
        # results utility owns the canonical mnist_results.csv append. Persist
        # only the selected QUBO head, using its actual post-selection test score.
        _persist_selected_qubo_result(
            seed=seed,
            test_accuracy=next(row["test_accuracy_pct"] for row in rows
                               if row["head"] == "QUBO Binary"),
            winner=tuning_winner,
            tuning_rows=tuning_rows,
            validation_fraction=args.validation_fraction,
            representation=representation,
            representation_path=representation_path,
            representation_source=representation_source,
            tuning_csv_path=tuning_path,
        )

    print("\nHead comparison (same frozen backbone/features):")
    print(f"{'Head':32s} {'Train %':>9s} {'Validation %':>14s} {'Test %':>9s}")
    for row in rows:
        print(f"{row['head']:32s} {row['train_accuracy_pct']:9.2f} {row['validation_accuracy_pct']:14.2f} {row['test_accuracy_pct']:9.2f}")
    print(f"Selected STE epoch by validation accuracy: {ste['epoch']}")
    print(f"CSV: {output_path.resolve()}")


if __name__ == "__main__":
    main()
