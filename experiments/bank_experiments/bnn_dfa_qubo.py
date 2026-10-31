"""Experiment 4: BNN + DFA + QUBO Standalone Individual Pipeline for Bank Dataset."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import torch
import torch.nn.functional as F

from experiments.bank_experiments.bank_representation import (
    collect_hidden_features,
    train_bank_backbone,
)
from experiments.bank_experiments.results_utils import (
    append_bank_result,
    Timer,
    count_parameters,
)
from experiments.bank_experiments.tune_qubo_validation import (
    calculate_final_scale_and_bias,
    create_binary_targets,
    evaluate_binary_head,
    tune_from_representations,
)
from models.bank_bnn import BankBNN
from models.qubo_head import calculate_ridge_ls_scales, optimize_binary_classifier
from utils.bank.data import NUM_CLASSES, NUM_FEATURES, get_bank_datasets, get_bank_loaders
from utils.bank.metrics import compute_classification_metrics
from utils.bank.seed import set_seed


def run_bnn_dfa_qubo(
    seed: int = 42,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    hidden_size: int = 64,
    num_reads: int = 50,
    num_sweeps: int = 500,
    tune_validation: bool = False,
    device_str: str | None = None,
):
    if device_str is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device_str)

    set_seed(seed)
    print(f"[Experiment 4: BNN + DFA + QUBO] Running with seed {seed} on {device}")

    train_ds, val_ds, test_ds = get_bank_datasets()

    with Timer() as overall_timer:
        # Step 1: Train BNN-DFA Backbone
        print("Training BNN-DFA backbone...")
        model, backbone_time = train_bank_backbone(
            train_ds, seed=seed, epochs=epochs, batch_size=batch_size, lr=lr,
            hidden_size=hidden_size, device=device,
        )

        for param in model.parameters():
            param.requires_grad = False

        # Step 2: Extract hidden features
        print("Extracting frozen hidden representations...")
        feat_start = time.perf_counter()
        tr_loader, val_loader, te_loader = get_bank_loaders(batch_size=batch_size)
        H_train, y_train = collect_hidden_features(model, tr_loader, device)
        H_val, y_val = collect_hidden_features(model, val_loader, device)
        H_test, y_test = collect_hidden_features(model, te_loader, device)
        feat_time = time.perf_counter() - feat_start

        # Step 3: QUBO Head Optimization
        print("Optimizing QUBO binary classifier heads...")
        qubo_start = time.perf_counter()

        if tune_validation:
            print("Performing validation grid search for QUBO hyperparameters...")
            win_row, best_w, best_sc, best_b, _, qubo_time = tune_from_representations(
                H_train, y_train, H_val, y_val, seed=seed,
            )
            selected_reads = win_row["num_reads"]
            selected_sweeps = win_row["num_sweeps"]
            selected_mult = win_row["scale_multiplier"]
            val_acc = win_row["validation_accuracy"]
            val_ba = win_row["validation_balanced_accuracy"]
        else:
            base_scales = calculate_ridge_ls_scales(H_train, y_train, num_classes=NUM_CLASSES)
            targets_train = create_binary_targets(y_train, num_classes=NUM_CLASSES)
            w_bin = optimize_binary_classifier(
                H=H_train,
                targets=targets_train,
                scales=base_scales,
                num_classes=NUM_CLASSES,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                seed=seed,
            )

            w_list, sc_list, b_list = [], [], []
            for class_id in range(NUM_CLASSES):
                w_refit, sc, b = calculate_final_scale_and_bias(
                    H_train, targets_train[:, class_id], w_bin[:, class_id],
                )
                w_list.append(w_refit)
                sc_list.append(sc)
                b_list.append(b)

            best_w = torch.stack(w_list, dim=1)
            best_sc = torch.stack(sc_list)
            best_b = torch.stack(b_list)
            qubo_time = time.perf_counter() - qubo_start

            val_scores = evaluate_binary_head(H_val, best_w, best_sc, best_b)
            val_preds = val_scores.argmax(dim=1).numpy()
            val_probs = torch.softmax(val_scores, dim=1)[:, 1].numpy()
            val_metrics = compute_classification_metrics(y_val.numpy(), val_preds, val_probs)
            val_acc = val_metrics["accuracy"]
            val_ba = val_metrics["balanced_accuracy"]
            selected_reads = num_reads
            selected_sweeps = num_sweeps
            selected_mult = 1.0

        # Step 4: Final Test Evaluation
        test_scores = evaluate_binary_head(H_test, best_w, best_sc, best_b)
        test_preds = test_scores.argmax(dim=1).numpy()
        test_probs = torch.softmax(test_scores, dim=1)[:, 1].numpy()
        test_metrics = compute_classification_metrics(y_test.numpy(), test_preds, test_probs)

    print(
        f"\nFinal Test Results (BNN + DFA + QUBO): "
        f"Acc: {test_metrics['accuracy']:.2f}% | BA: {test_metrics['balanced_accuracy']:.2f}% | "
        f"F1: {test_metrics['f1']:.2f}% | ROC-AUC: {test_metrics.get('roc_auc', 0):.2f}% | PR-AUC: {test_metrics.get('pr_auc', 0):.2f}%"
    )

    backbone_params = count_parameters(model.fc1)
    head_params = hidden_size * NUM_CLASSES + NUM_CLASSES * 2  # weights + scales + biases
    total_params = backbone_params + head_params

    append_bank_result(
        experiment="bnn_dfa_qubo",
        method="BNN + DFA + QUBO",
        seed=seed,
        test_accuracy=test_metrics["accuracy"],
        balanced_accuracy=test_metrics["balanced_accuracy"],
        f1_score=test_metrics["f1"],
        precision=test_metrics["precision"],
        recall=test_metrics["recall"],
        roc_auc=test_metrics.get("roc_auc"),
        pr_auc=test_metrics.get("pr_auc"),
        training_time_sec=overall_timer.seconds,
        qubo_time_sec=qubo_time,
        backbone_training_time_sec=backbone_time,
        feature_extraction_time_sec=feat_time,
        head_fit_time_sec=qubo_time,
        end_to_end_training_time_sec=overall_timer.seconds,
        current_run_total_time_sec=overall_timer.seconds,
        backbone_parameter_count=backbone_params,
        head_parameter_count=head_params,
        total_parameter_count=total_params,
        qubo_tuning_enabled=tune_validation,
        qubo_num_reads_selected=selected_reads,
        qubo_num_sweeps_selected=selected_sweeps,
        qubo_scale_multiplier_selected=selected_mult,
        qubo_validation_accuracy=val_acc,
        qubo_validation_balanced_accuracy=val_ba,
    )
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 4: BNN + DFA + QUBO for Bank Dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-reads", type=int, default=50)
    parser.add_argument("--num-sweeps", type=int, default=500)
    parser.add_argument("--tune-validation", action="store_true")
    args = parser.parse_args()

    run_bnn_dfa_qubo(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        tune_validation=args.tune_validation,
    )
