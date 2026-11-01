"""CIFAR-10 BNN+DFA with QUBO Binary Head.

Optimizes binary weights using simulated annealing on an Ising objective,
followed by class-specific continuous scale and bias fitting.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import torch

from experiments.CIFAR10_experiments.cifar_representation import (
    load_or_train_cifar10_representation,
)
from experiments.CIFAR10_experiments.results_utils import (
    append_result,
)
from experiments.CIFAR10_experiments.tune_qubo_validation import (
    DEFAULT_NUM_READS,
    DEFAULT_NUM_SWEEPS,
    DEFAULT_SCALE_MULTIPLIERS,
    _parse_grid,
    tune_from_representations,
    validate_grid,
    write_tuning_csv,
)
from experiments.MNIST_experiments.bnn_dfa_qubo import (
    calculate_final_scale_and_bias,
    create_binary_targets,
)
from models.qubo_head import (
    calculate_ridge_ls_scales,
    optimize_binary_classifier,
)
from utils.CIFAR10.data import get_device
from utils.CIFAR10.representation import DEFAULT_REPRESENTATION_DIR
from utils.MNIST.seed import set_seed


def fit_fixed_qubo(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    num_reads: int = 100,
    num_sweeps: int = 1000,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Optimize QUBO head with fixed annealing parameters."""
    t0 = time.perf_counter()
    targets = create_binary_targets(y_train, num_classes=10)
    scales = calculate_ridge_ls_scales(H_train, y_train, num_classes=10)
    scales = torch.where(scales > 1e-8, scales, torch.full_like(scales, 1e-8))

    w = optimize_binary_classifier(
        H=H_train,
        targets=targets,
        scales=scales,
        num_classes=10,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
    )
    w, final_scales, biases = calculate_final_scale_and_bias(
        H=H_train,
        labels=y_train,
        binary_weights=w,
        num_classes=10,
    )
    fit_time = time.perf_counter() - t0
    return w, final_scales, biases, fit_time


def evaluate_qubo(
    H: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
) -> float:
    scores = H.float() @ (w * scales.unsqueeze(0)) + biases
    return float((scores.argmax(dim=1) == y).float().mean().item() * 100.0)


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 BNN+DFA QUBO binary head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--representation-dir",
        type=Path,
        default=DEFAULT_REPRESENTATION_DIR,
    )
    parser.add_argument("--qubo-tuning", action="store_true", help="enable validation grid tuning")
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument(
        "--num-reads-grid",
        type=lambda s: _parse_grid(s, int, "num-reads-grid"),
        default=DEFAULT_NUM_READS,
    )
    parser.add_argument(
        "--num-sweeps-grid",
        type=lambda s: _parse_grid(s, int, "num-sweeps-grid"),
        default=DEFAULT_NUM_SWEEPS,
    )
    parser.add_argument(
        "--scale-multipliers",
        type=lambda s: _parse_grid(s, float, "scale-multipliers"),
        default=DEFAULT_SCALE_MULTIPLIERS,
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

    tuning_rows = None
    winner = None

    if args.qubo_tuning:
        validate_grid(args.num_reads_grid, args.num_sweeps_grid, args.scale_multipliers)
        winner, tuning_rows = tune_from_representations(
            H_train, y_train, H_val, y_val, args.seed,
            args.num_reads_grid, args.num_sweeps_grid, args.scale_multipliers,
        )
        tuning_csv_path = write_tuning_csv(
            rows=tuning_rows,
            seed=args.seed,
            epochs=args.epochs,
            validation_fraction=args.validation_fraction,
            output_dir=Path(os.environ.get("CIFAR10_QUBO_TUNING_DIR", "data/results_diagnostic/cifar10_qubo")),
            representation_artifact_id=representation.get("artifact_id", ""),
            n_train=len(y_train),
            n_validation=len(y_val),
            n_test=len(y_test),
        )
        print(f"[QUBO-TUNE] Saved candidate tuning CSV to {tuning_csv_path}")

        if winner is None:
            raise RuntimeError("All QUBO tuning candidates failed")
        w_bin, scales, biases = winner["_weights"], winner["_scales"], winner["_bias"]
        fit_time = float(winner["tuning_total_seconds"])
        qubo_method = "QUBO Binary (validation-tuned)"
        notes = (
            f"selected: reads={winner['num_reads']}, sweeps={winner['num_sweeps']}, "
            f"scale_mult={winner['scale_multiplier']}"
        )
    else:
        w_bin, scales, biases, fit_time = fit_fixed_qubo(
            H_train, y_train, num_reads=args.num_reads, num_sweeps=args.num_sweeps, seed=args.seed
        )
        qubo_method = "QUBO Binary (fixed config)"
        notes = f"fixed: reads={args.num_reads}, sweeps={args.num_sweeps}"

    train_acc = evaluate_qubo(H_train, y_train, w_bin, scales, biases)
    val_acc = evaluate_qubo(H_val, y_val, w_bin, scales, biases)
    test_acc = evaluate_qubo(H_test, y_test, w_bin, scales, biases)

    print(f"\n{qubo_method} Results:")
    print(f"Train Accuracy:      {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy:       {test_acc:.2f}%")
    print(f"Fit time:            {fit_time:.4f}s")

    metadata = representation["metadata"]
    backbone_params = metadata.get("backbone_parameter_count", 29002)
    head_params = int(w_bin.numel() + scales.numel() + biases.numel())  # 1280 + 10 + 10 = 1300

    append_result(
        experiment="cifar_dfa_qubo",
        method=qubo_method,
        seed=args.seed,
        test_accuracy=test_acc,
        validation_accuracy=val_acc,
        training_accuracy=train_acc,
        parameter_count=backbone_params + head_params,
        backbone_training_time_sec=representation.get("training_seconds", 0.0),
        feature_extraction_time_sec=representation.get("feature_extraction_seconds", 0.0),
        head_fit_time_sec=fit_time,
        backbone_parameter_count=backbone_params,
        head_parameter_count=head_params,
        total_parameter_count=backbone_params + head_params,
        pipeline_timing_status="complete",
        qubo_time_sec=fit_time,
        qubo_selected_candidate_time_sec=float(winner["selected_candidate_runtime_sec"]) if (winner and winner.get("selected_candidate_runtime_sec")) else None,
        qubo_tuning_enabled=args.qubo_tuning,
        qubo_num_reads_selected=winner["num_reads"] if winner else args.num_reads,
        qubo_num_sweeps_selected=winner["num_sweeps"] if winner else args.num_sweeps,
        qubo_scale_multiplier_selected=float(winner["scale_multiplier"]) if winner else None,
        qubo_solver_seed_selected=winner["solver_seed"] if winner else None,
        qubo_validation_accuracy=winner["validation_accuracy"] if winner else val_acc,
        qubo_validation_fraction=args.validation_fraction,
        qubo_grid_size=len(tuning_rows) if tuning_rows else 1,
        qubo_selection_metric="validation_accuracy" if args.qubo_tuning else "",
        representation_artifact_id=representation["artifact_id"],
        representation_path=str(rep_path.resolve()),
        representation_cache_status=rep_source,
        n_train=len(y_train),
        n_validation=len(y_val),
        n_test=len(y_test),
        dataset_fingerprint=metadata.get("dataset_fingerprint", ""),
        split_protocol=metadata.get("split_protocol", ""),
        representation_sha256=representation.get("representation_sha256", ""),
        notes=notes,
    )


if __name__ == "__main__":
    main()
