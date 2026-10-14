"""Validate the numerically stable LS head on MNIST.

Trains BNN+DFA once per seed, selects among SVD/ridge heads using a stratified
validation split of training representations, then evaluates the selected head
and the legacy default LS head on the test set for a one-time comparison.

No existing experiment or CSV is modified. Output: data/results_diagnostic/
ls_solver_validation_<timestamp>.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from models.bnn import BNN
from models.dfa import DFAClassifier
from models.ls_heads import (
    accuracy_from_scores,
    fit_ridge_classifier,
    predict_scores,
    select_ridge_lambda,
)
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


def fit_legacy_default(H: torch.Tensor, y: torch.Tensor):
    """Reproduce original float32/default-driver LS fit; returned weights on CPU."""
    H32 = H.detach().to(device="cpu", dtype=torch.float32)
    y = y.detach().to(device="cpu", dtype=torch.long)
    X = torch.cat(
        [H32, torch.ones((len(H32), 1), dtype=torch.float32)], dim=1
    )
    Y = F.one_hot(y, num_classes=NUM_CLASSES).float()
    result = torch.linalg.lstsq(X, Y)
    rank = int(result.rank.max()) if result.rank.numel() else -1
    return result.solution, rank


def stratified_split(
    labels: torch.Tensor, validation_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic per-class split, using training labels only."""
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
    # Shuffle pooled indices so class blocks do not create an artificial order.
    fit_idx = fit_idx[torch.randperm(len(fit_idx), generator=generator)]
    val_idx = val_idx[torch.randperm(len(val_idx), generator=generator)]
    return fit_idx, val_idx


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_seed(seed: int, args) -> list[dict]:
    device = pick_device()
    print(f"\n=== seed={seed} device={device} ===", flush=True)

    # Match the repository experiment's setup order.
    set_seed(seed)
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
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

    native_loss, native_test_acc = evaluate_dfa_model(
        model, test_loader, criterion, device
    )
    H_train, y_train = collect_hidden_features(model, train_loader, device)
    H_test, y_test = collect_hidden_features(model, test_loader, device)
    H_train, y_train = H_train.cpu(), y_train.cpu()
    H_test, y_test = H_test.cpu(), y_test.cpu()

    fit_idx, val_idx = stratified_split(
        y_train, args.validation_fraction, seed
    )
    H_fit, y_fit = H_train[fit_idx], y_train[fit_idx]
    H_val, y_val = H_train[val_idx], y_train[val_idx]

    rows = []
    candidates = []

    # Legacy LS is measured as a baseline, but deliberately excluded from
    # the final candidate selection because this test is evaluating a stable fix.
    try:
        W_legacy_fit, legacy_rank = fit_legacy_default(H_fit, y_fit)
        legacy_val = accuracy_from_scores(predict_scores(H_val, W_legacy_fit), y_val)
        candidates.append({
            "seed": seed, "device": str(device), "torch_version": torch.__version__,
            "epochs": args.epochs, "split": "stratified training-only",
            "candidate": "legacy_float32_default", "lambda_relative": "",
            "reported_rank": legacy_rank, "validation_accuracy": legacy_val,
            "test_accuracy": "", "role": "baseline_only", "error": "",
        })
        print(
            f"legacy float32/default LS: validation={legacy_val:.2f}% "
            f"rank={legacy_rank}",
            flush=True,
        )
    except Exception as exc:
        legacy_val = float("nan")
        W_legacy_fit = None
        legacy_rank = ""
        candidates.append({
            "seed": seed, "device": str(device), "torch_version": torch.__version__,
            "epochs": args.epochs, "split": "stratified training-only",
            "candidate": "legacy_float32_default", "lambda_relative": "",
            "reported_rank": "", "validation_accuracy": "",
            "test_accuracy": "", "role": "baseline_only",
            "error": f"{type(exc).__name__}: {exc}",
        })
        print(f"legacy LS failed: {type(exc).__name__}: {exc}", flush=True)

    # Unregularized SVD LS and ridge variants are stable candidates.
    stable_scores: dict[str, float] = {}
    try:
        t0 = time.perf_counter()
        W_svd = fit_ridge_classifier(H_fit, y_fit, NUM_CLASSES, lambda_relative=0.0)
        svd_val = accuracy_from_scores(predict_scores(H_val, W_svd), y_val)
        stable_scores["svd_gelsd_rcond_1e-6"] = svd_val
        candidates.append({
            "seed": seed, "device": str(device), "torch_version": torch.__version__,
            "epochs": args.epochs, "split": "stratified training-only",
            "candidate": "svd_gelsd_rcond_1e-6", "lambda_relative": 0,
            "reported_rank": "", "validation_accuracy": svd_val,
            "test_accuracy": "", "role": "eligible_for_selection",
            "error": "", "fit_seconds": round(time.perf_counter() - t0, 3),
        })
        print(f"SVD LS rcond=1e-6: validation={svd_val:.2f}%", flush=True)
    except Exception as exc:
        candidates.append({
            "seed": seed, "device": str(device), "torch_version": torch.__version__,
            "epochs": args.epochs, "split": "stratified training-only",
            "candidate": "svd_gelsd_rcond_1e-6", "lambda_relative": 0,
            "reported_rank": "", "validation_accuracy": "",
            "test_accuracy": "", "role": "eligible_for_selection",
            "error": f"{type(exc).__name__}: {exc}",
        })

    for lam in RIDGE_LAMBDAS:
        name = f"ridge_{lam:g}"
        t0 = time.perf_counter()
        try:
            W = fit_ridge_classifier(H_fit, y_fit, NUM_CLASSES, lam)
            val_acc = accuracy_from_scores(predict_scores(H_val, W), y_val)
            stable_scores[name] = val_acc
            candidates.append({
                "seed": seed, "device": str(device), "torch_version": torch.__version__,
                "epochs": args.epochs, "split": "stratified training-only",
                "candidate": name, "lambda_relative": lam,
                "reported_rank": "", "validation_accuracy": val_acc,
                "test_accuracy": "", "role": "eligible_for_selection",
                "error": "", "fit_seconds": round(time.perf_counter() - t0, 3),
            })
            print(f"{name}: validation={val_acc:.2f}%", flush=True)
        except Exception as exc:
            candidates.append({
                "seed": seed, "device": str(device), "torch_version": torch.__version__,
                "epochs": args.epochs, "split": "stratified training-only",
                "candidate": name, "lambda_relative": lam,
                "reported_rank": "", "validation_accuracy": "",
                "test_accuracy": "", "role": "eligible_for_selection",
                "error": f"{type(exc).__name__}: {exc}",
            })

    if not stable_scores:
        raise RuntimeError("No stable LS candidate succeeded. See CSV/error output.")

    # Choose only among SVD/ridge candidates, using validation accuracy.
    # Exact ties prefer stronger ridge, then weaker ridge, then unregularized SVD.
    tie_order = {
        "ridge_0.0001": 0,
        "ridge_1e-06": 1,
        "ridge_1e-08": 2,
        "svd_gelsd_rcond_1e-6": 3,
    }
    selected_name = sorted(
        stable_scores,
        key=lambda name: (-stable_scores[name], tie_order.get(name, 99)),
    )[0]
    if selected_name == "svd_gelsd_rcond_1e-6":
        selected_lambda = 0.0
    else:
        selected_lambda = float(selected_name.removeprefix("ridge_"))

    print(
        f"Selected using validation only: {selected_name} "
        f"({stable_scores[selected_name]:.2f}%)",
        flush=True,
    )

    # Refit both the legacy baseline and selected stable method on all training
    # representations, then evaluate each once on the held-out test set.
    final_rows = []
    if W_legacy_fit is not None:
        W_legacy_all, _ = fit_legacy_default(H_train, y_train)
        legacy_test = accuracy_from_scores(predict_scores(H_test, W_legacy_all), y_test)
        final_rows.append({
            "seed": seed, "device": str(device), "torch_version": torch.__version__,
            "epochs": args.epochs, "split": "full train refit after validation",
            "candidate": "legacy_float32_default_full_train",
            "lambda_relative": "", "reported_rank": legacy_rank,
            "validation_accuracy": legacy_val, "test_accuracy": legacy_test,
            "role": "baseline_test_comparison", "error": "",
            "native_bnn_dfa_test_accuracy": native_test_acc,
            "dfa_train_seconds": round(dfa_seconds, 2),
        })
        print(f"legacy LS test accuracy: {legacy_test:.2f}%", flush=True)

    W_selected = fit_ridge_classifier(
        H_train, y_train, NUM_CLASSES, lambda_relative=selected_lambda
    )
    selected_test = accuracy_from_scores(predict_scores(H_test, W_selected), y_test)
    final_rows.append({
        "seed": seed, "device": str(device), "torch_version": torch.__version__,
        "epochs": args.epochs, "split": "full train refit after validation",
        "candidate": f"selected_stable:{selected_name}",
        "lambda_relative": selected_lambda, "reported_rank": "",
        "validation_accuracy": stable_scores[selected_name],
        "test_accuracy": selected_test, "role": "selected_test_result", "error": "",
        "native_bnn_dfa_test_accuracy": native_test_acc,
        "dfa_train_seconds": round(dfa_seconds, 2),
    })
    print(f"selected stable LS test accuracy: {selected_test:.2f}%", flush=True)
    print(f"native BNN+DFA test accuracy: {native_test_acc:.2f}%", flush=True)

    # Normalize columns so all records append cleanly to one CSV schema.
    all_rows = candidates + final_rows
    fields = [
        "seed", "device", "torch_version", "epochs", "split", "candidate",
        "lambda_relative", "reported_rank", "validation_accuracy", "test_accuracy",
        "role", "error", "fit_seconds", "native_bnn_dfa_test_accuracy",
        "dfa_train_seconds",
    ]
    normalized = [{key: row.get(key, "") for key in fields} for row in all_rows]
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[314159])
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.05 <= args.validation_fraction <= 0.4:
        parser.error("--validation-fraction must be between 0.05 and 0.4")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"ls_solver_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, args))
    append_rows(output, rows)
    print(f"\nFinished. Results written to: {output}", flush=True)


if __name__ == "__main__":
    main()
