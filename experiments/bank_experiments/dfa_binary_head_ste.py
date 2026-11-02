"""Experiment 7: Straight-Through Estimator (STE) Binary Head on Bank BNN+DFA Representations."""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.bank_experiments.bank_representation import (
    load_or_train_bank_representation,
)
from experiments.bank_experiments.results_utils import (
    append_bank_result,
    Timer,
)
from experiments.bank_experiments.tune_qubo_validation import evaluate_binary_head
from models.ls_heads import fit_ridge_classifier
from utils.bank.data import NUM_CLASSES, get_bank_datasets
from utils.bank.metrics import compute_classification_metrics, find_best_threshold
from utils.bank.representation import DEFAULT_REPRESENTATION_DIR
from utils.bank.seed import set_seed


class STEBinaryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        return torch.where(input_tensor >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STEBinaryClassifier(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.latent_weight = nn.Parameter(torch.empty(in_features, out_features))
        self.scale = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_bin = STEBinaryFunction.apply(self.latent_weight)
        scaled_w = w_bin * self.scale.unsqueeze(0)
        return x @ scaled_w + self.bias.unsqueeze(0)


def train_ste_head(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    H_val: torch.Tensor,
    y_val: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    steps: int = 300,
    lr: float = 0.05,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Train STE binary head using training split, selecting best checkpoint by validation balanced accuracy."""
    set_seed(seed)
    hidden_dim = H_train.shape[1]
    model = STEBinaryClassifier(hidden_dim, num_classes)

    # Initialize from continuous Ridge weights for fast, high-quality convergence
    with torch.no_grad():
        ridge_w = fit_ridge_classifier(H_train, y_train, num_classes=num_classes, lambda_relative=1e-3)
        model.latent_weight.copy_(ridge_w[:-1, :].float())
        model.bias.copy_(ridge_w[-1, :].float())

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_ba = -1.0
    best_state = None
    best_thresh = 0.5

    H_tr = H_train.float()
    y_tr = y_train.long()
    H_v = H_val.float()
    y_v = y_val.long()

    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        out = model(H_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()

        # Validation check
        model.eval()
        with torch.no_grad():
            v_scores = model(H_v)
            v_probs = torch.softmax(v_scores, dim=1)[:, 1].numpy()
            cur_thresh = find_best_threshold(y_v.numpy(), v_probs)
            v_preds = (v_probs >= cur_thresh).astype(int)
            v_metrics = compute_classification_metrics(y_v.numpy(), v_preds, v_probs)
            v_ba = v_metrics["balanced_accuracy"]

            if v_ba > best_val_ba:
                best_val_ba = v_ba
                best_thresh = cur_thresh
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    final_binary_w = torch.where(model.latent_weight >= 0, 1.0, -1.0).detach()
    final_scales = model.scale.detach()
    final_biases = model.bias.detach()

    return final_binary_w, final_scales, final_biases, best_thresh


def run_dfa_ste_binary(
    seed: int = 42,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    hidden_size: int = 64,
    ste_steps: int = 300,
    ste_lr: float = 0.05,
    representation_dir: Path | str = DEFAULT_REPRESENTATION_DIR,
    device_str: str | None = None,
):
    if device_str is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str

    set_seed(seed)
    print(f"\n{'='*70}")
    print(f"[Experiment 7: STE Binary Head] Seed: {seed} | Epochs: {epochs}")
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

        print("\nTraining STE Binary Head with validation checkpointing...")
        fit_start = time.perf_counter()
        w_bin, scales, biases, best_threshold = train_ste_head(
            H_train, y_train, H_val, y_val,
            num_classes=NUM_CLASSES, steps=ste_steps, lr=ste_lr, seed=seed,
        )
        head_fit_time = time.perf_counter() - fit_start

        test_scores = evaluate_binary_head(H_test, w_bin, scales, biases)
        test_probs = torch.softmax(test_scores, dim=1)[:, 1].numpy()
        test_preds = (test_probs >= best_threshold).astype(int)
        metrics = compute_classification_metrics(y_test.numpy(), test_preds, test_probs)

        print(
            f"STE Binary Head (Threshold {best_threshold:.4f}) -> Acc: {metrics['accuracy']:.2f}% | "
            f"BA: {metrics['balanced_accuracy']:.2f}% | "
            f"F1: {metrics['f1']:.2f}% | ROC-AUC: {metrics.get('roc_auc', 0):.2f}%"
        )

        backbone_params = (hidden_dim * 66 + hidden_dim) + (hidden_dim * NUM_CLASSES + NUM_CLASSES)
        head_params = int(w_bin.numel() + scales.numel() + biases.numel())
        total_time = backbone_time + feat_time + head_fit_time

        append_bank_result(
            experiment="dfa_binary_head_ste",
            method="STE Binary Head",
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
    parser = argparse.ArgumentParser(description="Experiment 7: STE Binary Head on Bank Dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--ste-lr", type=float, default=0.05)
    args = parser.parse_args()

    run_dfa_ste_binary(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        ste_steps=args.steps,
        ste_lr=args.ste_lr,
    )
