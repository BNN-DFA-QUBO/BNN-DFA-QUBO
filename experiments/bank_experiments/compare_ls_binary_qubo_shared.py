"""Experiment 5: Shared Frozen Representation Head Comparison on Bank Dataset."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from experiments.bank_experiments.bank_representation import (
    load_or_train_bank_representation,
)
from experiments.bank_experiments.results_utils import (
    append_bank_result,
    Timer,
)
from experiments.bank_experiments.tune_qubo_validation import (
    DEFAULT_NUM_READS,
    DEFAULT_NUM_SWEEPS,
    DEFAULT_SCALE_MULTIPLIERS,
    _parse_grid,
    evaluate_binary_head,
    tune_from_representations,
    validate_grid,
)
from models.ls_heads import fit_ridge_classifier, predict_scores
from utils.bank.data import NUM_CLASSES, get_bank_datasets
from utils.bank.metrics import compute_classification_metrics
from utils.bank.representation import DEFAULT_REPRESENTATION_DIR
from utils.bank.seed import set_seed


def fit_binarized_ridge(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    ridge_weights: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bipolarize Ridge feature weights and refit continuous scale & bias on training rows."""
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

        if scale_raw.item() < 0:
            binary_weights[:, class_id] *= -1
            scale = -scale_raw
        else:
            scale = scale_raw
        scales.append(scale)
        biases.append(target_mean - scale_raw * z_mean)

    return binary_weights, torch.stack(scales), torch.stack(biases)


def run_shared_comparison(
    seed: int = 42,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    hidden_size: int = 64,
    lambda_relative: float = 1e-4,
    num_reads_grid=DEFAULT_NUM_READS,
    num_sweeps_grid=DEFAULT_NUM_SWEEPS,
    scale_multipliers=DEFAULT_SCALE_MULTIPLIERS,
    representation_dir: Path | str = DEFAULT_REPRESENTATION_DIR,
    device_str: str | None = None,
) -> dict[str, dict[str, Any]]:
    if device_str is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str

    set_seed(seed)
    print(f"\n{'='*70}")
    print(f"[Experiment 5: Shared Head Comparison] Seed: {seed} | Epochs: {epochs}")
    print(f"{'='*70}")

    train_ds, val_ds, test_ds = get_bank_datasets()

    with Timer() as total_timer:
        # Step 1: Extract/load ONE shared frozen representation
        payload, repr_path, cache_status, backbone_time, feat_time = load_or_train_bank_representation(
            train_ds, val_ds, test_ds, seed=seed, epochs=epochs, batch_size=batch_size,
            lr=lr, hidden_size=hidden_size, device=device, output_dir=representation_dir,
        )

        H_train = payload["H_train"]
        y_train = payload["y_train"]
        H_val = payload["H_val"]
        y_val = payload["y_val"]
        H_test = payload["H_test"]
        y_test = payload["y_test"]

        n_train, hidden_dim = H_train.shape
        n_val = H_val.shape[0]
        n_test = H_test.shape[0]
        print(f"Representations loaded ({cache_status}): H_train: {H_train.shape}, H_val: {H_val.shape}, H_test: {H_test.shape}")

        results: dict[str, dict[str, Any]] = {}

        # -------------------------------------------------------------
        # Head 1: Real Ridge LS Head (continuous upper bound on frozen repr)
        # -------------------------------------------------------------
        print("\n--- Fitting Head 1: Real Ridge LS ---")
        ridge_start = time.perf_counter()
        ridge_weights = fit_ridge_classifier(
            H_train, y_train, num_classes=NUM_CLASSES, lambda_relative=lambda_relative,
        )
        ridge_time = time.perf_counter() - ridge_start

        test_scores_ridge = predict_scores(H_test, ridge_weights)
        test_preds_ridge = test_scores_ridge.argmax(dim=1).numpy()
        test_probs_ridge = torch.softmax(test_scores_ridge, dim=1)[:, 1].numpy()
        metrics_ridge = compute_classification_metrics(y_test.numpy(), test_preds_ridge, test_probs_ridge)

        print(
            f"Real Ridge LS -> Acc: {metrics_ridge['accuracy']:.2f}% | BA: {metrics_ridge['balanced_accuracy']:.2f}% | "
            f"F1: {metrics_ridge['f1']:.2f}% | ROC-AUC: {metrics_ridge.get('roc_auc', 0):.2f}%"
        )
        results["Real Ridge LS"] = metrics_ridge

        append_bank_result(
            experiment="compare_ls_binary_qubo_shared",
            method="Real Ridge LS Head",
            seed=seed,
            test_accuracy=metrics_ridge["accuracy"],
            balanced_accuracy=metrics_ridge["balanced_accuracy"],
            f1_score=metrics_ridge["f1"],
            precision=metrics_ridge["precision"],
            recall=metrics_ridge["recall"],
            roc_auc=metrics_ridge.get("roc_auc"),
            pr_auc=metrics_ridge.get("pr_auc"),
            training_time_sec=ridge_time,
            head_fit_time_sec=ridge_time,
            backbone_training_time_sec=backbone_time,
            feature_extraction_time_sec=feat_time,
            end_to_end_training_time_sec=backbone_time + feat_time + ridge_time,
            current_run_total_time_sec=ridge_time,
            representation_path=str(repr_path),
            representation_cache_status=cache_status,
            n_train=n_train,
            n_validation=n_val,
            n_test=n_test,
        )

        # -------------------------------------------------------------
        # Head 2: Binarized Ridge LS Head (sign-binarized + refit scale & bias)
        # -------------------------------------------------------------
        print("\n--- Fitting Head 2: Binarized Ridge LS (Refit Scale+Bias) ---")
        bin_ridge_start = time.perf_counter()
        w_bin_ridge, sc_bin_ridge, b_bin_ridge = fit_binarized_ridge(
            H_train, y_train, ridge_weights, num_classes=NUM_CLASSES,
        )
        bin_ridge_time = time.perf_counter() - bin_ridge_start

        test_scores_bin_ridge = evaluate_binary_head(H_test, w_bin_ridge, sc_bin_ridge, b_bin_ridge)
        test_preds_bin_ridge = test_scores_bin_ridge.argmax(dim=1).numpy()
        test_probs_bin_ridge = torch.softmax(test_scores_bin_ridge, dim=1)[:, 1].numpy()
        metrics_bin_ridge = compute_classification_metrics(y_test.numpy(), test_preds_bin_ridge, test_probs_bin_ridge)

        print(
            f"Binarized Ridge LS -> Acc: {metrics_bin_ridge['accuracy']:.2f}% | BA: {metrics_bin_ridge['balanced_accuracy']:.2f}% | "
            f"F1: {metrics_bin_ridge['f1']:.2f}% | ROC-AUC: {metrics_bin_ridge.get('roc_auc', 0):.2f}%"
        )
        results["Binarized Ridge LS"] = metrics_bin_ridge

        append_bank_result(
            experiment="compare_ls_binary_qubo_shared",
            method="Binarized Ridge LS (Refit Scale+Bias)",
            seed=seed,
            test_accuracy=metrics_bin_ridge["accuracy"],
            balanced_accuracy=metrics_bin_ridge["balanced_accuracy"],
            f1_score=metrics_bin_ridge["f1"],
            precision=metrics_bin_ridge["precision"],
            recall=metrics_bin_ridge["recall"],
            roc_auc=metrics_bin_ridge.get("roc_auc"),
            pr_auc=metrics_bin_ridge.get("pr_auc"),
            training_time_sec=bin_ridge_time,
            head_fit_time_sec=bin_ridge_time,
            backbone_training_time_sec=backbone_time,
            feature_extraction_time_sec=feat_time,
            end_to_end_training_time_sec=backbone_time + feat_time + bin_ridge_time,
            current_run_total_time_sec=bin_ridge_time,
            representation_path=str(repr_path),
            representation_cache_status=cache_status,
            n_train=n_train,
            n_validation=n_val,
            n_test=n_test,
        )

        # -------------------------------------------------------------
        # Head 3: QUBO Binary Head (tuned on validation representation only)
        # -------------------------------------------------------------
        print("\n--- Tuning & Fitting Head 3: QUBO Binary Head ---")
        win_row, best_w_qubo, best_sc_qubo, best_b_qubo, tuning_rows, qubo_time = tune_from_representations(
            H_train=H_train,
            y_train=y_train,
            H_val=H_val,
            y_val=y_val,
            seed=seed,
            num_reads_grid=num_reads_grid,
            num_sweeps_grid=num_sweeps_grid,
            scale_multipliers=scale_multipliers,
        )
        print(f"Selected QUBO config from validation: reads={win_row['num_reads']}, sweeps={win_row['num_sweeps']}, mult={win_row['scale_multiplier']}")

        test_scores_qubo = evaluate_binary_head(H_test, best_w_qubo, best_sc_qubo, best_b_qubo)
        test_preds_qubo = test_scores_qubo.argmax(dim=1).numpy()
        test_probs_qubo = torch.softmax(test_scores_qubo, dim=1)[:, 1].numpy()
        metrics_qubo = compute_classification_metrics(y_test.numpy(), test_preds_qubo, test_probs_qubo)

        print(
            f"QUBO Binary Head -> Acc: {metrics_qubo['accuracy']:.2f}% | BA: {metrics_qubo['balanced_accuracy']:.2f}% | "
            f"F1: {metrics_qubo['f1']:.2f}% | ROC-AUC: {metrics_qubo.get('roc_auc', 0):.2f}%"
        )
        results["QUBO Binary Head"] = metrics_qubo

        append_bank_result(
            experiment="compare_ls_binary_qubo_shared",
            method="QUBO Binary Head",
            seed=seed,
            test_accuracy=metrics_qubo["accuracy"],
            balanced_accuracy=metrics_qubo["balanced_accuracy"],
            f1_score=metrics_qubo["f1"],
            precision=metrics_qubo["precision"],
            recall=metrics_qubo["recall"],
            roc_auc=metrics_qubo.get("roc_auc"),
            pr_auc=metrics_qubo.get("pr_auc"),
            training_time_sec=qubo_time,
            head_fit_time_sec=qubo_time,
            qubo_time_sec=qubo_time,
            backbone_training_time_sec=backbone_time,
            feature_extraction_time_sec=feat_time,
            end_to_end_training_time_sec=backbone_time + feat_time + qubo_time,
            current_run_total_time_sec=qubo_time,
            qubo_tuning_enabled=True,
            qubo_num_reads_selected=win_row["num_reads"],
            qubo_num_sweeps_selected=win_row["num_sweeps"],
            qubo_scale_multiplier_selected=win_row["scale_multiplier"],
            qubo_validation_accuracy=win_row["validation_accuracy"],
            qubo_validation_balanced_accuracy=win_row.get("validation_balanced_accuracy"),
            qubo_grid_size=len(tuning_rows),
            qubo_solver_seed_selected=win_row["solver_seed"],
            qubo_selection_metric="validation_balanced_accuracy",
            representation_path=str(repr_path),
            representation_cache_status=cache_status,
            n_train=n_train,
            n_validation=n_val,
            n_test=n_test,
        )

    print(f"\n{'='*70}")
    print("Shared Head Comparison Summary on Identical Frozen Representations:")
    print(f"{'Method':<35} | {'Acc (%)':<8} | {'BA (%)':<8} | {'F1 (%)':<8} | {'ROC-AUC':<8}")
    print("-" * 75)
    for name, m in results.items():
        print(f"{name:<35} | {m['accuracy']:<8.2f} | {m['balanced_accuracy']:<8.2f} | {m['f1']:<8.2f} | {m.get('roc_auc', 0):<8.2f}")
    print(f"{'='*70}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 5: Shared Representation Head Comparison for Bank Dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-reads-grid", type=lambda s: _parse_grid(s, int, "num-reads-grid"), default=DEFAULT_NUM_READS)
    parser.add_argument("--num-sweeps-grid", type=lambda s: _parse_grid(s, int, "num-sweeps-grid"), default=DEFAULT_NUM_SWEEPS)
    parser.add_argument("--scale-multipliers", type=lambda s: _parse_grid(s, float, "scale-multipliers"), default=DEFAULT_SCALE_MULTIPLIERS)
    args = parser.parse_args()

    run_shared_comparison(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_reads_grid=args.num_reads_grid,
        num_sweeps_grid=args.num_sweeps_grid,
        scale_multipliers=args.scale_multipliers,
    )

