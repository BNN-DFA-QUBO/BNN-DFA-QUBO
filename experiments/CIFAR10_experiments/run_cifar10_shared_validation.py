"""Canonical CIFAR-10 shared-representation head comparison.

Evaluates all five classifier heads on the same frozen BNN+DFA representation:
1. Real Least Squares (Ridge LS)
2. Binarized Least Squares
3. Direct Binary
4. STE Binary (validation-only checkpointing)
5. QUBO Binary (validation-tuned or fixed)

Rules:
- All heads share the identical representation artifact for a given seed.
- Model selection and hyperparameter tuning use validation data ONLY.
- Official test partition is evaluated only once for final reporting.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any, Dict, List

import torch

from experiments.CIFAR10_experiments.cifar_dfa_binarized_ls import (
    evaluate_binarized_head,
)
from experiments.CIFAR10_experiments.cifar_dfa_direct_binary import (
    evaluate_direct_binary_head,
)
from experiments.CIFAR10_experiments.cifar_dfa_least_squares import (
    fit_and_evaluate_ridge,
)
from experiments.CIFAR10_experiments.cifar_dfa_qubo import (
    evaluate_qubo,
    fit_fixed_qubo,
)
from experiments.CIFAR10_experiments.cifar_dfa_ste_binary import (
    evaluate_ste_head,
)
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
from utils.CIFAR10.data import get_device
from utils.CIFAR10.representation import DEFAULT_REPRESENTATION_DIR
from utils.MNIST.seed import set_seed


def run_cifar10_shared_validation(
    seed: int = 42,
    epochs: int = 20,
    ste_epochs: int = 50,
    direct_steps: int = 200,
    validation_fraction: float = 0.1,
    representation_dir: Path | str = DEFAULT_REPRESENTATION_DIR,
    fixed_qubo: bool = False,
    num_reads: int = 100,
    num_sweeps: int = 1000,
    num_reads_grid: tuple = DEFAULT_NUM_READS,
    num_sweeps_grid: tuple = DEFAULT_NUM_SWEEPS,
    scale_multipliers: tuple = DEFAULT_SCALE_MULTIPLIERS,
    device: str | torch.device | None = None,
    train_dataset: Any = None,
    test_dataset: Any = None,
) -> List[Dict[str, Any]]:
    """Execute the full 5-head comparison on a shared representation."""
    if device is None:
        device = get_device()
    set_seed(seed)
    print(f"\n=======================================================")
    print(f"CIFAR-10 Shared-Representation Experiment | Seed: {seed}")
    print(f"Device: {device}")
    print(f"=======================================================\n")

    representation, rep_path, rep_source = load_or_train_cifar10_representation(
        seed=seed,
        epochs=epochs,
        validation_fraction=validation_fraction,
        device=device,
        output_dir=representation_dir,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )
    H_train, y_train = representation["H_train"], representation["y_train"]
    H_val, y_val = representation["H_validation"], representation["y_validation"]
    H_test, y_test = representation["H_test"], representation["y_test"]

    metadata = representation["metadata"]
    artifact_id = representation["artifact_id"]
    backbone_params = metadata.get("backbone_parameter_count", 29002)
    training_time = float(representation.get("training_seconds", 0.0))
    extraction_time = float(representation.get("feature_extraction_seconds", 0.0))

    results = []

    # ---------------------------------------------------------
    # 1. Real Least Squares (Ridge LS)
    # ---------------------------------------------------------
    print("--- 1. Real Least Squares (Ridge LS) ---")
    ridge_w, ridge_accs, ridge_fit_time = fit_and_evaluate_ridge(
        H_train, y_train, H_val, y_val, H_test, y_test, lambda_relative=1e-4
    )
    ridge_params = int(ridge_w.numel())
    append_result(
        experiment="cifar_shared_validation",
        method="Real Least Squares",
        seed=seed,
        test_accuracy=ridge_accs["test"],
        validation_accuracy=ridge_accs["validation"],
        training_accuracy=ridge_accs["train"],
        parameter_count=backbone_params + ridge_params,
        backbone_training_time_sec=training_time,
        feature_extraction_time_sec=extraction_time,
        head_fit_time_sec=ridge_fit_time,
        current_run_total_time_sec=training_time + extraction_time + ridge_fit_time,
        backbone_parameter_count=backbone_params,
        head_parameter_count=ridge_params,
        total_parameter_count=backbone_params + ridge_params,
        pipeline_timing_status="complete",
        representation_artifact_id=artifact_id,
        representation_path=str(rep_path.resolve()),
        representation_cache_status=rep_source,
        n_train=len(y_train),
        n_validation=len(y_val),
        n_test=len(y_test),
        dataset_fingerprint=metadata.get("dataset_fingerprint", ""),
        split_protocol=metadata.get("split_protocol", ""),
        representation_sha256=representation.get("representation_sha256", ""),
    )
    results.append({
        "head": "Real Least Squares",
        "train": ridge_accs["train"],
        "val": ridge_accs["validation"],
        "test": ridge_accs["test"],
        "fit_time": ridge_fit_time,
        "head_params": ridge_params,
    })

    # ---------------------------------------------------------
    # 2. Binarized Least Squares
    # ---------------------------------------------------------
    print("--- 2. Binarized Least Squares ---")
    (bin_w, bin_scale, bin_bias), bin_accs, bin_fit_time = evaluate_binarized_head(
        H_train, y_train, H_val, y_val, H_test, y_test, ridge_w
    )
    bin_params = int(bin_w.numel() + bin_scale.numel() + bin_bias.numel())
    append_result(
        experiment="cifar_shared_validation",
        method="Binarized Least Squares",
        seed=seed,
        test_accuracy=bin_accs["test"],
        validation_accuracy=bin_accs["validation"],
        training_accuracy=bin_accs["train"],
        parameter_count=backbone_params + bin_params,
        backbone_training_time_sec=training_time,
        feature_extraction_time_sec=extraction_time,
        head_fit_time_sec=bin_fit_time,
        current_run_total_time_sec=training_time + extraction_time + bin_fit_time,
        backbone_parameter_count=backbone_params,
        head_parameter_count=bin_params,
        total_parameter_count=backbone_params + bin_params,
        pipeline_timing_status="complete",
        representation_artifact_id=artifact_id,
        representation_path=str(rep_path.resolve()),
        representation_cache_status=rep_source,
        n_train=len(y_train),
        n_validation=len(y_val),
        n_test=len(y_test),
        dataset_fingerprint=metadata.get("dataset_fingerprint", ""),
        split_protocol=metadata.get("split_protocol", ""),
        representation_sha256=representation.get("representation_sha256", ""),
    )
    results.append({
        "head": "Binarized Least Squares",
        "train": bin_accs["train"],
        "val": bin_accs["validation"],
        "test": bin_accs["test"],
        "fit_time": bin_fit_time,
        "head_params": bin_params,
    })

    # ---------------------------------------------------------
    # 3. Direct Binary
    # ---------------------------------------------------------
    print("--- 3. Direct Binary ---")
    (dir_w, dir_bias), dir_accs, dir_fit_time = evaluate_direct_binary_head(
        H_train, y_train, H_val, y_val, H_test, y_test,
        steps=direct_steps, seed=seed,
    )
    dir_params = int(dir_w.numel() + dir_bias.numel())
    append_result(
        experiment="cifar_shared_validation",
        method="Direct Binary",
        seed=seed,
        test_accuracy=dir_accs["test"],
        validation_accuracy=dir_accs["validation"],
        training_accuracy=dir_accs["train"],
        parameter_count=backbone_params + dir_params,
        backbone_training_time_sec=training_time,
        feature_extraction_time_sec=extraction_time,
        head_fit_time_sec=dir_fit_time,
        current_run_total_time_sec=training_time + extraction_time + dir_fit_time,
        backbone_parameter_count=backbone_params,
        head_parameter_count=dir_params,
        total_parameter_count=backbone_params + dir_params,
        pipeline_timing_status="complete",
        representation_artifact_id=artifact_id,
        representation_path=str(rep_path.resolve()),
        representation_cache_status=rep_source,
        n_train=len(y_train),
        n_validation=len(y_val),
        n_test=len(y_test),
        dataset_fingerprint=metadata.get("dataset_fingerprint", ""),
        split_protocol=metadata.get("split_protocol", ""),
        representation_sha256=representation.get("representation_sha256", ""),
    )
    results.append({
        "head": "Direct Binary",
        "train": dir_accs["train"],
        "val": dir_accs["validation"],
        "test": dir_accs["test"],
        "fit_time": dir_fit_time,
        "head_params": dir_params,
    })

    # ---------------------------------------------------------
    # 4. STE Binary
    # ---------------------------------------------------------
    print("--- 4. STE Binary ---")
    ste_best, ste_accs, ste_fit_time = evaluate_ste_head(
        H_train, y_train, H_val, y_val, H_test, y_test,
        epochs=ste_epochs, seed=seed,
    )
    ste_params = int(ste_best["weights"].numel() + ste_best["bias"].numel())
    append_result(
        experiment="cifar_shared_validation",
        method="STE Binary",
        seed=seed,
        test_accuracy=ste_accs["test"],
        validation_accuracy=ste_accs["validation"],
        training_accuracy=ste_accs["train"],
        parameter_count=backbone_params + ste_params,
        backbone_training_time_sec=training_time,
        feature_extraction_time_sec=extraction_time,
        head_fit_time_sec=ste_fit_time,
        current_run_total_time_sec=training_time + extraction_time + ste_fit_time,
        backbone_parameter_count=backbone_params,
        head_parameter_count=ste_params,
        total_parameter_count=backbone_params + ste_params,
        pipeline_timing_status="complete",
        representation_artifact_id=artifact_id,
        representation_path=str(rep_path.resolve()),
        representation_cache_status=rep_source,
        n_train=len(y_train),
        n_validation=len(y_val),
        n_test=len(y_test),
        dataset_fingerprint=metadata.get("dataset_fingerprint", ""),
        split_protocol=metadata.get("split_protocol", ""),
        representation_sha256=representation.get("representation_sha256", ""),
        notes=f"best_val_epoch={ste_best['epoch']}",
    )
    results.append({
        "head": "STE Binary",
        "train": ste_accs["train"],
        "val": ste_accs["validation"],
        "test": ste_accs["test"],
        "fit_time": ste_fit_time,
        "head_params": ste_params,
    })

    # ---------------------------------------------------------
    # 5. QUBO Binary
    # ---------------------------------------------------------
    print("--- 5. QUBO Binary ---")
    winner = None
    rows = None
    if fixed_qubo:
        qubo_w, qubo_scale, qubo_bias, qubo_fit_time = fit_fixed_qubo(
            H_train, y_train, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed
        )
        qubo_val = evaluate_qubo(H_val, y_val, qubo_w, qubo_scale, qubo_bias)
        qubo_test = evaluate_qubo(H_test, y_test, qubo_w, qubo_scale, qubo_bias)
        qubo_train = evaluate_qubo(H_train, y_train, qubo_w, qubo_scale, qubo_bias)
        qubo_config_note = f"fixed reads={num_reads}, sweeps={num_sweeps}"
    else:
        winner, rows = tune_from_representations(
            H_train, y_train, H_val, y_val, seed,
            num_reads_grid, num_sweeps_grid, scale_multipliers,
        )
        tuning_csv_path = write_tuning_csv(
            rows=rows,
            seed=seed,
            epochs=epochs,
            validation_fraction=validation_fraction,
            output_dir=Path(os.environ.get("CIFAR10_QUBO_TUNING_DIR", "data/results_diagnostic/cifar10_qubo")),
            representation_artifact_id=artifact_id,
            n_train=len(y_train),
            n_validation=len(y_val),
            n_test=len(y_test),
        )
        print(f"[QUBO-TUNE] Saved candidate tuning CSV to {tuning_csv_path}")
        if winner is None:
            raise RuntimeError("QUBO tuning yielded no valid solutions")
        qubo_w = winner["_weights"]
        qubo_scale = winner["_scales"]
        qubo_bias = winner["_bias"]
        qubo_fit_time = float(winner["tuning_total_seconds"])
        qubo_val = winner["validation_accuracy"]
        qubo_train = evaluate_qubo(H_train, y_train, qubo_w, qubo_scale, qubo_bias)
        qubo_test = evaluate_qubo(H_test, y_test, qubo_w, qubo_scale, qubo_bias)
        qubo_config_note = (
            f"selected: reads={winner['num_reads']}, sweeps={winner['num_sweeps']}, "
            f"scale_mult={winner['scale_multiplier']}"
        )

    qubo_params = int(qubo_w.numel() + qubo_scale.numel() + qubo_bias.numel())
    append_result(
        experiment="cifar_shared_validation",
        method="QUBO Binary",
        seed=seed,
        test_accuracy=qubo_test,
        validation_accuracy=qubo_val,
        training_accuracy=qubo_train,
        parameter_count=backbone_params + qubo_params,
        backbone_training_time_sec=training_time,
        feature_extraction_time_sec=extraction_time,
        head_fit_time_sec=qubo_fit_time,
        current_run_total_time_sec=training_time + extraction_time + qubo_fit_time,
        backbone_parameter_count=backbone_params,
        head_parameter_count=qubo_params,
        total_parameter_count=backbone_params + qubo_params,
        pipeline_timing_status="complete",
        qubo_time_sec=qubo_fit_time,
        qubo_selected_candidate_time_sec=float(winner["selected_candidate_runtime_sec"]) if (winner and winner.get("selected_candidate_runtime_sec")) else None,
        qubo_tuning_enabled=not fixed_qubo,
        qubo_num_reads_selected=winner["num_reads"] if winner else num_reads,
        qubo_num_sweeps_selected=winner["num_sweeps"] if winner else num_sweeps,
        qubo_scale_multiplier_selected=float(winner["scale_multiplier"]) if winner else None,
        qubo_solver_seed_selected=winner["solver_seed"] if winner else None,
        qubo_validation_accuracy=qubo_val,
        qubo_validation_fraction=validation_fraction,
        qubo_grid_size=len(rows) if not fixed_qubo and rows else 1,
        qubo_selection_metric="validation_accuracy" if not fixed_qubo else "",
        representation_artifact_id=artifact_id,
        representation_path=str(rep_path.resolve()),
        representation_cache_status=rep_source,
        n_train=len(y_train),
        n_validation=len(y_val),
        n_test=len(y_test),
        dataset_fingerprint=metadata.get("dataset_fingerprint", ""),
        split_protocol=metadata.get("split_protocol", ""),
        representation_sha256=representation.get("representation_sha256", ""),
        notes=qubo_config_note,
    )
    results.append({
        "head": "QUBO Binary",
        "train": qubo_train,
        "val": qubo_val,
        "test": qubo_test,
        "fit_time": qubo_fit_time,
        "head_params": qubo_params,
    })

    # Summary Display
    print("\n" + "=" * 76)
    print(f"{'Method':<26} | {'Train %':<8} | {'Val %':<8} | {'Test %':<8} | {'Time (s)':<8}")
    print("-" * 76)
    for r in results:
        print(f"{r['head']:<26} | {r['train']:<8.2f} | {r['val']:<8.2f} | {r['test']:<8.2f} | {r['fit_time']:<8.2f}")
    print("=" * 76 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 5-head comparison on shared representation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ste-epochs", type=int, default=50)
    parser.add_argument("--direct-steps", type=int, default=200)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--representation-dir",
        type=Path,
        default=DEFAULT_REPRESENTATION_DIR,
    )
    parser.add_argument("--fixed-qubo", action="store_true")
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

    run_cifar10_shared_validation(
        seed=args.seed,
        epochs=args.epochs,
        ste_epochs=args.ste_epochs,
        direct_steps=args.direct_steps,
        validation_fraction=args.validation_fraction,
        representation_dir=args.representation_dir,
        fixed_qubo=args.fixed_qubo,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        num_reads_grid=args.num_reads_grid,
        num_sweeps_grid=args.num_sweeps_grid,
        scale_multipliers=args.scale_multipliers,
    )


if __name__ == "__main__":
    main()
