"""Straight-through-estimator binary head for saved Stroke DFA representations.

This version standardizes hidden units using training rows only, normalizes
logit scale by sqrt(hidden_size), and compares several training-only
initializations by validation metrics. Test labels are used only once to report
metrics for the selected validation candidate.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from models.ls_heads import fit_ridge_regression
from utils.stroke_representation import load_stroke_head_representation
from utils.stroke_binary_features import standardize_hidden_features
from utils.stroke_metrics import find_best_threshold, calculate_metrics
from utils.seed import set_seed
from utils.stroke_config import STE_EPOCHS, STE_LEARNING_RATE, STE_PATIENCE, STE_MIN_DELTA


def _binary_sign(values, fallback=None):
    signs = torch.sign(values)
    if fallback is not None:
        signs = torch.where(signs == 0, torch.sign(fallback), signs)
    return torch.where(signs == 0, torch.ones_like(signs), signs)


def _initial_direction(name, H_train, y_train, seed):
    """Return a real-valued, train-only direction before binary sign projection."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if name == "random":
        return torch.randn(H_train.shape[1], generator=generator)
    if name == "class_mean":
        return H_train[y_train == 1].mean(0) - H_train[y_train == 0].mean(0)
    if name == "ridge":
        coefficients = fit_ridge_regression(
            H_train, y_train, lambda_relative=1e-3
        )[:-1, 0].to(dtype=torch.float32)
        return coefficients
    raise ValueError(f"Unknown STE initialization: {name}")


def _initial_signs(name, H_train, y_train, seed):
    direction = _initial_direction(name, H_train, y_train, seed)
    fallback = None
    if name == "ridge":
        fallback = H_train[y_train == 1].mean(0) - H_train[y_train == 0].mean(0)
    return _binary_sign(direction, fallback)


def _confidence_weighted_latent_start(direction, signs, seed):
    """Make uncertain latent signs easier to flip while preserving initial signs.

    STE's straight-through gradient updates latent real weights, but the forward
    pass only observes sign(latent_weights). Initializing every magnitude to 0.5
    can trap the optimizer in its starting binary vector. This start uses the
    continuous training-derived direction as a confidence score: weak direction
    components begin near zero, while stronger components begin farther away.
    """
    direction = torch.as_tensor(direction, dtype=torch.float32).detach()
    signs = torch.as_tensor(signs, dtype=torch.float32).detach()
    max_abs = direction.abs().max().clamp_min(1e-8)
    confidence = (direction.abs() / max_abs).clamp(0.0, 1.0)
    magnitudes = 0.025 + 0.175 * confidence
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 19)
    jitter = 0.001 * torch.randn(signs.shape, generator=generator)
    latent = signs * magnitudes + jitter
    # Jitter is deliberately too small to change the selected starting signs.
    latent = torch.where(torch.sign(latent) == 0, signs * 0.025, latent)
    return latent


def _fit_candidate(H_train, y_train, H_val, y_val, seed, initialization):
    """Train an STE start, retaining its best validation checkpoint including epoch zero.

    The binary initialization is itself a valid classifier. Evaluating it before
    optimization prevents STE updates from replacing a useful sign pattern with
    a degenerate late-epoch pattern (for example, scores that predict every case
    positive). Checkpoint and stopping decisions use validation data only.
    """
    set_seed(seed)
    direction = _initial_direction(initialization, H_train, y_train, seed)
    fallback = None
    if initialization == "ridge":
        fallback = H_train[y_train == 1].mean(0) - H_train[y_train == 0].mean(0)
    signs = _binary_sign(direction, fallback)
    initial_binary_weights = signs.detach().clone()
    latent_weights = _confidence_weighted_latent_start(
        direction, signs, seed
    ).detach().requires_grad_(True)
    bias = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam(
        [latent_weights, bias],
        lr=STE_LEARNING_RATE,
    )
    logit_scale = H_train.shape[1] ** 0.5
    criterion = torch.nn.BCEWithLogitsLoss()
    best = None
    checkpoints = 0

    def consider_checkpoint(epoch):
        nonlocal best, checkpoints
        with torch.no_grad():
            weights = _binary_sign(latent_weights).detach().clone()
            val_scores = (H_val @ weights) / logit_scale + bias
        scores_numpy = val_scores.detach().cpu().numpy()
        threshold, validation_ba = find_best_threshold(
            y_val.numpy(), scores_numpy
        )
        validation_metrics = calculate_metrics(
            y_val.numpy(), scores_numpy, threshold
        )
        validation_selection_score = float(np.mean([
            validation_ba,
            validation_metrics["f1"],
            validation_metrics["pr_auc"],
            validation_metrics["roc_auc"],
        ]))
        selection_key = (
            validation_selection_score,
            float(validation_ba),
            float(validation_metrics["f1"]),
            float(validation_metrics["pr_auc"]),
            float(validation_metrics["roc_auc"]),
        )
        checkpoints += 1
        if best is None or selection_key > best["selection_key"]:
            best = {
                "selection_key": selection_key,
                "validation_selection_score": validation_selection_score,
                "weights": weights,
                "bias": float(bias.detach().item()),
                "threshold": float(threshold),
                "validation_balanced_accuracy": float(validation_ba),
                "validation_f1": float(validation_metrics["f1"]),
                "validation_pr_auc": float(validation_metrics["pr_auc"]),
                "validation_roc_auc": float(validation_metrics["roc_auc"]),
                "initialization": initialization,
                "best_epoch": int(epoch),
            }
        return validation_selection_score

    # Keep the initialized sign pattern as a baseline candidate. With the
    # class-mean start, its ranking matches the Direct Binary baseline.
    best_score_for_patience = consider_checkpoint(0)
    last_significant_improvement = 0
    epochs_run = 0
    patience = max(int(STE_PATIENCE), 1)

    for epoch in range(1, int(STE_EPOCHS) + 1):
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
        epochs_run = epoch

        current_score = consider_checkpoint(epoch)
        if current_score > best_score_for_patience + STE_MIN_DELTA:
            best_score_for_patience = current_score
            last_significant_improvement = epoch
        if epoch - last_significant_improvement >= patience:
            break

    with torch.no_grad():
        final_weights = _binary_sign(latent_weights).detach()
    best["epochs_run"] = int(epochs_run)
    best["validation_checkpoint_count"] = int(checkpoints)
    best["weight_flips_from_initial"] = int(
        (best["weights"] != initial_binary_weights).sum().item()
    )
    best["final_weight_flips_from_initial"] = int(
        (final_weights != initial_binary_weights).sum().item()
    )
    best["initial_binary_weights"] = initial_binary_weights
    return best


def train_ste_binary(representation_path, seed):
    """Train three STE starts; select by equal-weight validation BA/F1/PR-AUC/ROC-AUC composite."""
    data = load_stroke_head_representation(representation_path)
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
        "validation_selection_score": best["validation_selection_score"],
        "validation_f1": best["validation_f1"],
        "validation_pr_auc": best["validation_pr_auc"],
        "validation_roc_auc": best["validation_roc_auc"],
        "binary_weights": best["weights"].numpy(),
        "bias": best["bias"],
        "initialization": best["initialization"],
        "candidate_initializations": len(candidates),
        "ste_epochs": STE_EPOCHS,
        "ste_epochs_run": best["epochs_run"],
        "ste_best_epoch": best["best_epoch"],
        "ste_patience": STE_PATIENCE,
        "ste_validation_checkpoints": best["validation_checkpoint_count"],
        "ste_weight_flips_from_init": best["weight_flips_from_initial"],
        "ste_final_weight_flips_from_init": best["final_weight_flips_from_initial"],
        "ste_latent_init": "confidence_weighted_train_direction",
        "ste_learning_rate": STE_LEARNING_RATE,
        "training_loss": "BCEWithLogitsLoss on standardized hidden features with logits divided by sqrt(hidden_size); epoch-zero baseline plus validation-selected checkpoint",
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
