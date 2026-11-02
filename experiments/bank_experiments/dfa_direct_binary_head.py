"""Experiment 6: Direct Binary Head on Bank BNN+DFA Representations."""
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
from experiments.bank_experiments.tune_qubo_validation import evaluate_binary_head
from utils.bank.data import NUM_CLASSES, get_bank_datasets
from utils.bank.metrics import compute_classification_metrics, find_best_threshold
from utils.bank.representation import DEFAULT_REPRESENTATION_DIR
from utils.bank.seed import set_seed


def fit_direct_binary(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit direct binary weights (+1/-1) via class centroid difference + refit scale & bias."""
    H = H_train.float()
    y = y_train.long()
    hidden_dim = H.shape[1]

    binary_weights = torch.zeros((hidden_dim, num_classes), dtype=torch.float32)
    scales = []
    biases = []

    for class_id in range(num_classes):
        pos_mask = (y == class_id)
        neg_mask = ~pos_mask

        pos_mean = H[pos_mask].mean(dim=0) if pos_mask.sum() > 0 else torch.zeros(hidden_dim)
        neg_mean = H[neg_mask].mean(dim=0) if neg_mask.sum() > 0 else torch.zeros(hidden_dim)

        diff = pos_mean - neg_mean
        w = torch.where(diff >= 0, 1.0, -1.0)
        binary_weights[:, class_id] = w

        target = torch.where(pos_mask, 1.0, -1.0).float()
        z = H @ w
        z_mean, t_mean = z.mean(), target.mean()
        z_centered, t_centered = z - z_mean, target - t_mean
        denom = z_centered @ z_centered

        if denom.item() > 1e-12:
            scale_raw = (z_centered @ t_centered) / denom
        else:
            scale_raw = torch.tensor(1.0, dtype=z.dtype)

        if scale_raw.item() < 0:
            binary_weights[:, class_id] *= -1
            scale = -scale_raw
        else:
            scale = scale_raw

        scales.append(scale)
        biases.append(t_mean - scale_raw * z_mean)

    return binary_weights, torch.stack(scales), torch.stack(biases)


def run_dfa_direct_binary(
    seed: int = 42,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    hidden_size: int = 64,
    representation_dir: Path | str = DEFAULT_REPRESENTATION_DIR,
    device_str: str | None = None,
):
    if device_str is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str

    set_seed(seed)
    print(f"\n{'='*70}")
    print(f"[Experiment 6: Direct Binary Head] Seed: {seed} | Epochs: {epochs}")
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

        print("\nFitting Direct Binary Head via class-centroid difference...")
        fit_start = time.perf_counter()
        w_bin, scales, biases = fit_direct_binary(H_train, y_train, num_classes=NUM_CLASSES)
        head_fit_time = time.perf_counter() - fit_start

        # Threshold tuning on validation representations
        val_scores = evaluate_binary_head(H_val, w_bin, scales, biases)
        val_probs = torch.softmax(val_scores, dim=1)[:, 1].numpy()
        best_threshold = find_best_threshold(y_val.numpy(), val_probs)

        # Test evaluation with selected threshold
        test_scores = evaluate_binary_head(H_test, w_bin, scales, biases)
        test_probs = torch.softmax(test_scores, dim=1)[:, 1].numpy()
        test_preds = (test_probs >= best_threshold).astype(int)
        metrics = compute_classification_metrics(y_test.numpy(), test_preds, test_probs)

        print(
            f"Direct Binary Head (Threshold {best_threshold:.4f}) -> Acc: {metrics['accuracy']:.2f}% | "
            f"BA: {metrics['balanced_accuracy']:.2f}% | "
            f"F1: {metrics['f1']:.2f}% | ROC-AUC: {metrics.get('roc_auc', 0):.2f}%"
        )

        backbone_params = (hidden_dim * 66 + hidden_dim) + (hidden_dim * NUM_CLASSES + NUM_CLASSES)
        head_params = int(w_bin.numel() + scales.numel() + biases.numel())
        total_time = backbone_time + feat_time + head_fit_time

        append_bank_result(
            experiment="dfa_direct_binary_head",
            method="Direct Binary Head",
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
    parser = argparse.ArgumentParser(description="Experiment 6: Direct Binary Head on Bank Dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    args = parser.parse_args()

    run_dfa_direct_binary(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
    )
