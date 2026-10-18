"""Straight-through-estimator binary head for saved Stroke DFA representations.

This version standardizes hidden units using training rows only, normalizes
logit scale by sqrt(hidden_size), and compares several training-only
initializations by validation metrics. Test labels are used only once to report
metrics for the selected validation candidate.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch

from models.ls_heads import fit_ridge_regression
from utils.stroke_representation import load_representation
from utils.stroke_binary_features import standardize_hidden_features
from utils.stroke_metrics import find_best_threshold, calculate_metrics
from utils.seed import set_seed
from utils.stroke_config import STE_EPOCHS, STE_LEARNING_RATE


def _binary_sign(values, fallback=None):
    signs = torch.sign(values)
    if fallback is not None:
        signs = torch.where(signs == 0, torch.sign(fallback), signs)
    return torch.where(signs == 0, torch.ones_like(signs), signs)


def _initial_signs(name, H_train, y_train, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if name == "random":
        return _binary_sign(torch.randn(H_train.shape[1], generator=generator))
    if name == "class_mean":
        direction = H_train[y_train == 1].mean(0) - H_train[y_train == 0].mean(0)
        return _binary_sign(direction)
    if name == "ridge":
        # Initialization uses training examples only; intercept is excluded.
        coefficients = fit_ridge_regression(
            H_train, y_train, lambda_relative=1e-3
        )[:-1, 0].to(dtype=torch.float32)
        direction = H_train[y_train == 1].mean(0) - H_train[y_train == 0].mean(0)
        return _binary_sign(coefficients, direction)
    raise ValueError(f"Unknown STE initialization: {name}")


def _fit_candidate(H_train, y_train, H_val, y_val, seed, initialization):
    set_seed(seed)
    signs = _initial_signs(initialization, H_train, y_train, seed)
    latent_weights = (0.5 * signs + 0.01 * torch.randn_like(signs)).detach().requires_grad_(True)
    bias = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam(
        [latent_weights, bias],
        lr=STE_LEARNING_RATE,
    )
    logit_scale = H_train.shape[1] ** 0.5
    criterion = torch.nn.BCEWithLogitsLoss()

    for _ in range(STE_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        sign_weights = _binary_sign(latent_weights)
        binary_weights = (
            sign_weights.detach()
            - latent_weights.detach()
            + latent_weights
        )
        logits = (H_train @ binary_weights) / logit_scale + bias
        loss = criterion(logits, y_train)
        if not torch.isfinite(loss):
            raise ValueError("STE training produced a non-finite loss")
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        weights = _binary_sign(latent_weights)
        val_scores = (H_val @ weights) / logit_scale + bias
    threshold, validation_ba = find_best_threshold(y_val.numpy(), val_scores.numpy())
    validation_metrics = calculate_metrics(y_val.numpy(), val_scores.numpy(), threshold)
    selection_key = (
        float(validation_ba),
        float(validation_metrics["f1"]),
        float(validation_metrics["pr_auc"]),
        float(validation_metrics["roc_auc"]),
    )
    return {
        "selection_key": selection_key,
        "weights": weights.detach().clone(),
        "bias": float(bias.detach().item()),
        "threshold": float(threshold),
        "validation_balanced_accuracy": float(validation_ba),
        "validation_f1": float(validation_metrics["f1"]),
        "validation_pr_auc": float(validation_metrics["pr_auc"]),
        "validation_roc_auc": float(validation_metrics["roc_auc"]),
        "initialization": initialization,
    }


def train_ste_binary(representation_path, seed):
    """Train three STE starts; rank by validation BA then F1, PR-AUC, and ROC-AUC."""
    data = load_representation(representation_path)
    H_train, H_val, H_test, transform = standardize_hidden_features(
        data["H_train"], data["H_val"], data["H_test"]
    )
    y_train = torch.as_tensor(data["y_train"]).float().reshape(-1)
    y_val = torch.as_tensor(data["y_val"]).float().reshape(-1)
    y_test = torch.as_tensor(data["y_test"]).float().reshape(-1)

    if set(torch.unique(y_train).tolist()) != {0.0, 1.0}:
        raise ValueError("STE training requires both binary classes")
    if set(torch.unique(y_val).tolist()) != {0.0, 1.0}:
        raise ValueError("STE validation requires both binary classes")

    candidates = []
    for index, initialization in enumerate(("ridge", "class_mean", "random")):
        candidate = _fit_candidate(
            H_train, y_train, H_val, y_val,
            seed=int(seed) + index * 1009,
            initialization=initialization,
        )
        candidates.append(candidate)
    best = max(candidates, key=lambda item: item["selection_key"])

    logit_scale = H_train.shape[1] ** 0.5
    test_scores = (
        H_test @ best["weights"]
    ) / logit_scale + best["bias"]
    metrics = calculate_metrics(y_test.numpy(), test_scores.numpy(), best["threshold"])

    return {
        "method": "STE Binary",
        "seed": int(data.get("training_seed", data.get("seed", seed))),
        "head_seed": int(seed),
        "threshold": best["threshold"],
        "validation_balanced_accuracy": best["validation_balanced_accuracy"],
        "validation_f1": best["validation_f1"],
        "validation_pr_auc": best["validation_pr_auc"],
        "validation_roc_auc": best["validation_roc_auc"],
        "binary_weights": best["weights"].numpy(),
        "bias": best["bias"],
        "initialization": best["initialization"],
        "candidate_initializations": len(candidates),
        "ste_epochs": STE_EPOCHS,
        "ste_learning_rate": STE_LEARNING_RATE,
        "training_loss": "BCEWithLogitsLoss on standardized hidden features with logits divided by sqrt(hidden_size)",
        "hidden_feature_transform": transform["transform"],
        "standardized_constant_units": transform["constant_units"],
        **metrics,
    }


if __name__ == "__main__":
    import argparse
    from utils.stroke_data import DATA_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", type=Path, default=DATA_DIR / "stroke_dfa_representation_seed_42.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_ste_binary(args.representation, args.seed)
    print("\nSTE Binary")
    print(f"Initialization: {result['initialization']}")
    print(f"Validation BA: {result['validation_balanced_accuracy'] * 100:.4f}%")
    print(f"Test BA: {result['balanced_accuracy'] * 100:.4f}%")
    print(f"F1: {result['f1'] * 100:.4f}%")
    print(f"ROC-AUC: {result['roc_auc'] * 100:.4f}%")
    print(f"PR-AUC: {result['pr_auc'] * 100:.4f}%")
    print(f"Threshold: {result['threshold']:.6f}")
