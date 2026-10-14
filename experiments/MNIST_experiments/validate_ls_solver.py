"""Validate a numerically stable MNIST LS head on an unseen seed.

The original MNIST training set is stratified into backbone-fit (80%) and
validation (20%) subsets BEFORE BNN+DFA training. Validation examples therefore
do not influence the learned representation. Solver selection uses validation
accuracy only. After selecting the solver/lambda, the classifier head is refit
using representations from all original MNIST training examples; test data are
never used to choose the configuration.

This diagnostic does not modify original experiment code or existing result
CSVs. The backbone itself trains on 80% of the standard MNIST training split,
so absolute results are not a replacement for the paper's full-training results.
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models.bnn import BNN
from models.dfa import DFAClassifier
from models.ls_heads import accuracy_from_scores, fit_ridge_classifier, predict_scores
from utils.MNIST.data import get_mnist_loaders
from utils.MNIST.seed import set_seed
from experiments.MNIST_experiments.dfa_least_squares import (
    train_dfa,
    evaluate_dfa_model,
    collect_hidden_features,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "results_diagnostic"
NUM_CLASSES = 10
RIDGE_LAMBDAS = (1e-8, 1e-6, 1e-4)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def stratified_split(
    labels: torch.Tensor, validation_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically split indices per class, before model training."""
    labels = labels.detach().cpu().long()
    generator = torch.Generator(device="cpu").manual_seed(seed + 100_003)
    fit_parts, val_parts = [], []
    for cls in torch.unique(labels):
        indices = torch.where(labels == cls)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n_val = max(1, int(round(len(indices) * validation_fraction)))
        val_parts.append(indices[:n_val])
        fit_parts.append(indices[n_val:])
    fit_idx = torch.cat(fit_parts)
    val_idx = torch.cat(val_parts)
    fit_idx = fit_idx[torch.randperm(len(fit_idx), generator=generator)]
    val_idx = val_idx[torch.randperm(len(val_idx), generator=generator)]
    return fit_idx, val_idx


def fit_legacy_default(H: torch.Tensor, y: torch.Tensor):
    """Original float32 torch.linalg.lstsq default-driver baseline."""
    H32 = H.detach().to(device="cpu", dtype=torch.float32)
    y = y.detach().to(device="cpu", dtype=torch.long)
    X = torch.cat(
        [H32, torch.ones((len(H32), 1), dtype=torch.float32)], dim=1
    )
    Y = F.one_hot(y, num_classes=NUM_CLASSES).float()
    result = torch.linalg.lstsq(X, Y)
    rank = int(result.rank.max()) if result.rank.numel() else -1
    return result.solution, rank


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed", "device", "torch_version", "epochs", "backbone_train_samples",
        "validation_samples", "head_fit_samples", "candidate", "lambda_relative", "reported_rank",
        "validation_accuracy", "test_accuracy", "role", "error",
        "fit_seconds", "native_bnn_dfa_test_accuracy", "dfa_train_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def run_seed(seed: int, args) -> list[dict]:
    device = pick_device()
    print(f"\n=== seed={seed} device={device} ===", flush=True)

    # Set seed before loaders/model, as the project does.
    set_seed(seed)
    full_train_loader, test_loader = get_mnist_loaders(batch_size=64)
    train_dataset = full_train_loader.dataset
    labels_all = getattr(train_dataset, "targets", None)
    if labels_all is None:
        raise RuntimeError("MNIST training dataset does not expose .targets for stratified split")

    fit_idx, val_idx = stratified_split(labels_all, args.validation_fraction, seed)
    train_loader = DataLoader(
        Subset(train_dataset, fit_idx.tolist()), batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        Subset(train_dataset, val_idx.tolist()), batch_size=64, shuffle=False
    )
    print(
        f"Backbone-fit examples: {len(fit_idx)}; held-out validation examples: {len(val_idx)}",
        flush=True,
    )

    model = BNN().to(device)
    dfa = DFAClassifier(hidden_size=128, output_size=NUM_CLASSES, device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start = time.perf_counter()
    for epoch in range(args.epochs):
        loss, train_acc = train_dfa(
            model=model, dfa=dfa, train_loader=train_loader,
            optimizer=optimizer, criterion=criterion, device=device,
        )
        print(
            f"epoch {epoch + 1}/{args.epochs}: loss={loss:.4f}, "
            f"train_acc={train_acc:.2f}%",
            flush=True,
        )
    dfa_seconds = time.perf_counter() - start

    _, native_test_acc = evaluate_dfa_model(model, test_loader, criterion, device)
    H_train, y_train = collect_hidden_features(model, train_loader, device)
    H_val, y_val = collect_hidden_features(model, val_loader, device)
    H_test, y_test = collect_hidden_features(model, test_loader, device)
    H_train, y_train = H_train.cpu(), y_train.cpu()
    H_val, y_val = H_val.cpu(), y_val.cpu()
    H_test, y_test = H_test.cpu(), y_test.cpu()

    candidates: list[dict] = []
    stable_scores: dict[str, float] = {}
    legacy_rank = ""
    legacy_fit = None

    # Legacy LS is a baseline only; it cannot win model selection.
    try:
        legacy_fit, legacy_rank = fit_legacy_default(H_train, y_train)
        legacy_val = accuracy_from_scores(predict_scores(H_val, legacy_fit), y_val)
        candidates.append({
            "seed": seed, "device": str(device), "torch_version": torch.__version__,
            "epochs": args.epochs, "backbone_train_samples": len(y_train),
            "validation_samples": len(y_val), "candidate": "legacy_float32_default",
            "lambda_relative": "", "reported_rank": legacy_rank,
            "validation_accuracy": legacy_val, "test_accuracy": "",
            "role": "baseline_only", "error": "",
        })
        print(
            f"legacy float32/default LS: validation={legacy_val:.2f}% rank={legacy_rank}",
            flush=True,
        )
    except Exception as exc:
        candidates.append({
            "seed": seed, "device": str(device), "torch_version": torch.__version__,
            "epochs": args.epochs, "backbone_train_samples": len(y_train),
            "validation_samples": len(y_val), "candidate": "legacy_float32_default",
            "lambda_relative": "", "reported_rank": "", "validation_accuracy": "",
            "test_accuracy": "", "role": "baseline_only",
            "error": f"{type(exc).__name__}: {exc}",
        })
        print(f"legacy LS failed: {type(exc).__name__}: {exc}", flush=True)

    # Compare the SVD LS candidate and ridge candidates on the same frozen H.
    try:
        t0 = time.perf_counter()
        W = fit_ridge_classifier(H_train, y_train, NUM_CLASSES, lambda_relative=0.0)
        val_acc = accuracy_from_scores(predict_scores(H_val, W), y_val)
        stable_scores["svd_gelsd_rcond_1e-6"] = val_acc
        candidates.append({
            "seed": seed, "device": str(device), "torch_version": torch.__version__,
            "epochs": args.epochs, "backbone_train_samples": len(y_train),
            "validation_samples": len(y_val), "candidate": "svd_gelsd_rcond_1e-6",
            "lambda_relative": 0.0, "reported_rank": "",
            "validation_accuracy": val_acc, "test_accuracy": "",
            "role": "eligible_for_selection", "error": "",
            "fit_seconds": round(time.perf_counter() - t0, 3),
        })
        print(f"SVD LS rcond=1e-6: validation={val_acc:.2f}%", flush=True)
    except Exception as exc:
        candidates.append({
            "seed": seed, "device": str(device), "torch_version": torch.__version__,
            "epochs": args.epochs, "backbone_train_samples": len(y_train),
            "validation_samples": len(y_val), "candidate": "svd_gelsd_rcond_1e-6",
            "lambda_relative": 0.0, "reported_rank": "", "validation_accuracy": "",
            "test_accuracy": "", "role": "eligible_for_selection",
            "error": f"{type(exc).__name__}: {exc}",
        })

    for lam in RIDGE_LAMBDAS:
        name = f"ridge_{lam:g}"
        t0 = time.perf_counter()
        try:
            W = fit_ridge_classifier(H_train, y_train, NUM_CLASSES, lam)
            val_acc = accuracy_from_scores(predict_scores(H_val, W), y_val)
            stable_scores[name] = val_acc
            candidates.append({
                "seed": seed, "device": str(device), "torch_version": torch.__version__,
                "epochs": args.epochs, "backbone_train_samples": len(y_train),
                "validation_samples": len(y_val), "candidate": name,
                "lambda_relative": lam, "reported_rank": "",
                "validation_accuracy": val_acc, "test_accuracy": "",
                "role": "eligible_for_selection", "error": "",
                "fit_seconds": round(time.perf_counter() - t0, 3),
            })
            print(f"{name}: validation={val_acc:.2f}%", flush=True)
        except Exception as exc:
            candidates.append({
                "seed": seed, "device": str(device), "torch_version": torch.__version__,
                "epochs": args.epochs, "backbone_train_samples": len(y_train),
                "validation_samples": len(y_val), "candidate": name,
                "lambda_relative": lam, "reported_rank": "",
                "validation_accuracy": "", "test_accuracy": "",
                "role": "eligible_for_selection",
                "error": f"{type(exc).__name__}: {exc}",
            })

    if not stable_scores:
        raise RuntimeError("No SVD/ridge candidate succeeded.")

    # Selection excludes the legacy baseline and uses only held-out validation.
    tie_order = {
        "ridge_0.0001": 0,
        "ridge_1e-06": 1,
        "ridge_1e-08": 2,
        "svd_gelsd_rcond_1e-6": 3,
    }
    selected_name = sorted(
        stable_scores, key=lambda name: (-stable_scores[name], tie_order.get(name, 99))
    )[0]
    selected_lambda = (
        0.0 if selected_name == "svd_gelsd_rcond_1e-6"
        else float(selected_name.removeprefix("ridge_"))
    )
    print(
        f"Selected using validation only: {selected_name} "
        f"({stable_scores[selected_name]:.2f}%)",
        flush=True,
    )

    # After selecting configuration on validation, refit the heads with all
    # original training examples (fit subset + held-out validation subset). The
    # backbone remains trained on fit subset only; test remains untouched.
    H_all = torch.cat([H_train, H_val], dim=0)
    y_all = torch.cat([y_train, y_val], dim=0)
    final_rows = []
    if legacy_fit is not None:
        W_legacy_all, _ = fit_legacy_default(H_all, y_all)
        legacy_test = accuracy_from_scores(predict_scores(H_test, W_legacy_all), y_test)
        final_rows.append({
            "seed": seed, "device": str(device), "torch_version": torch.__version__,
            "epochs": args.epochs, "backbone_train_samples": len(y_train),
            "validation_samples": len(y_val), "head_fit_samples": len(y_all),
            "candidate": "legacy_float32_default", "lambda_relative": "",
            "reported_rank": legacy_rank, "validation_accuracy": legacy_val,
            "test_accuracy": legacy_test, "role": "baseline_test_comparison",
            "error": "", "native_bnn_dfa_test_accuracy": native_test_acc,
            "dfa_train_seconds": round(dfa_seconds, 2),
        })
        print(f"legacy LS test accuracy: {legacy_test:.2f}%", flush=True)

    W_selected = fit_ridge_classifier(
        H_all, y_all, NUM_CLASSES, lambda_relative=selected_lambda
    )
    selected_test = accuracy_from_scores(predict_scores(H_test, W_selected), y_test)
    final_rows.append({
        "seed": seed, "device": str(device), "torch_version": torch.__version__,
        "epochs": args.epochs, "backbone_train_samples": len(y_train),
        "validation_samples": len(y_val), "head_fit_samples": len(y_all),
        "candidate": f"selected_stable:{selected_name}",
        "lambda_relative": selected_lambda, "reported_rank": "",
        "validation_accuracy": stable_scores[selected_name],
        "test_accuracy": selected_test, "role": "selected_test_result", "error": "",
        "native_bnn_dfa_test_accuracy": native_test_acc,
        "dfa_train_seconds": round(dfa_seconds, 2),
    })
    print(f"selected stable LS test accuracy: {selected_test:.2f}%", flush=True)
    print(f"native BNN+DFA test accuracy: {native_test_acc:.2f}%", flush=True)
    return candidates + final_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[314159])
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.05 <= args.validation_fraction <= 0.4:
        parser.error("--validation-fraction must be between 0.05 and 0.4")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / (
        f"ls_solver_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    rows = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, args))
    append_rows(output, rows)
    print(f"\nFinished. Results written to: {output}", flush=True)


if __name__ == "__main__":
    main()
