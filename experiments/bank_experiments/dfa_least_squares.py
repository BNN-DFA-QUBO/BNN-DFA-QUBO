"""Experiment 4: Real Ridge Least Squares Head on Bank BNN+DFA Representations."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np
import torch

from experiments.bank_experiments.bank_representation import (
    load_or_train_bank_representation,
)
from experiments.bank_experiments.results_utils import (
    append_bank_result,
    Timer,
)
from models.ls_heads import fit_ridge_classifier, predict_scores
from utils.bank.data import NUM_CLASSES, get_bank_datasets
from utils.bank.metrics import compute_classification_metrics, find_best_threshold
from utils.bank.representation import DEFAULT_REPRESENTATION_DIR
from utils.bank.seed import set_seed


def run_real_ls(
    seed: int = 42,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    hidden_size: int = 64,
    lambda_relative: float = 1e-4,
    representation_dir: Path | str = DEFAULT_REPRESENTATION_DIR,
    device_str: str | None = None,
):
    if device_str is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str

    set_seed(seed)
    print(f"\n{'='*70}")
    print(f"[Experiment 4: Real Ridge LS Head] Seed: {seed} | Epochs: {epochs}")
    print(f"{'='*70}")

    train_ds, val_ds, test_ds = get_bank_datasets()

    with Timer() as total_timer:
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

        print("\nFitting Continuous Ridge LS Head...")
        fit_start = time.perf_counter()
        ridge_weights = fit_ridge_classifier(
            H_train, y_train, num_classes=NUM_CLASSES, lambda_relative=lambda_relative,
        )
        head_fit_time = time.perf_counter() - fit_start

        # Threshold tuning on validation representations
        val_scores = predict_scores(H_val, ridge_weights)
        val_probs = torch.softmax(val_scores, dim=1)[:, 1].numpy()
        best_threshold = find_best_threshold(y_val.numpy(), val_probs)

        # Test evaluation with selected threshold
        test_scores = predict_scores(H_test, ridge_weights)
        test_probs = torch.softmax(test_scores, dim=1)[:, 1].numpy()
        test_preds = (test_probs >= best_threshold).astype(int)
        metrics = compute_classification_metrics(y_test.numpy(), test_preds, test_probs)

        print(
            f"Ridge LS Head (Threshold {best_threshold:.4f}) -> Acc: {metrics['accuracy']:.2f}% | "
            f"BA: {metrics['balanced_accuracy']:.2f}% | "
            f"F1: {metrics['f1']:.2f}% | ROC-AUC: {metrics.get('roc_auc', 0):.2f}%"
        )

        backbone_params = (hidden_dim * 66 + hidden_dim) + (hidden_dim * NUM_CLASSES + NUM_CLASSES)
        head_params = int(ridge_weights.numel())
        total_time = backbone_time + feat_time + head_fit_time

        append_bank_result(
            experiment="dfa_least_squares",
            method="Continuous Ridge Least Squares Head",
            seed=seed,
            test_accuracy=metrics["accuracy"],
            balanced_accuracy=metrics["balanced_accuracy"],
            f1_score=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            roc_auc=metrics.get("roc_auc"),
            pr_auc=metrics.get("pr_auc"),
            training_time_sec=total_time,
            head_fit_time_sec=head_fit_time,
            backbone_training_time_sec=backbone_time,
            feature_extraction_time_sec=feat_time,
            end_to_end_training_time_sec=total_time,
            current_run_total_time_sec=total_timer.elapsed,
            pipeline_timing_status="complete",
            backbone_parameter_count=backbone_params,
            head_parameter_count=head_params,
            total_parameter_count=backbone_params + head_params,
            representation_path=str(repr_path),
            representation_cache_status=cache_status,
            n_train=n_train,
            n_validation=n_val,
            n_test=n_test,
        )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 4: Real Ridge LS Head on Bank Dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lambda-relative", type=float, default=1e-4)
    args = parser.parse_args()

    run_real_ls(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        lambda_relative=args.lambda_relative,
    )
