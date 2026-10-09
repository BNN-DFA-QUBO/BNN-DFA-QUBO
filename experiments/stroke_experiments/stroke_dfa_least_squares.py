"""Signed-target least-squares head; unlike stroke_real_ls this fits targets {-1,+1}."""
import argparse
from pathlib import Path

import torch

from utils.stroke_representation import load_representation
from utils.stroke_data import DATA_DIR
from utils.stroke_metrics import find_best_threshold, calculate_metrics


def fit_signed_least_squares(hidden, labels):
    """Fit a {-1,+1} target with an explicit intercept via stable least squares."""
    design = torch.cat((hidden.float(), torch.ones(len(hidden), 1)), dim=1)
    signed_targets = (2.0 * labels.float() - 1.0).unsqueeze(1)
    solution = torch.linalg.lstsq(design, signed_targets).solution
    return solution[:, 0]


def train_signed_least_squares(representation_path):
    data = load_representation(representation_path)
    weights = fit_signed_least_squares(data["H_train"], data["y_train"])
    val_design = torch.cat((data["H_val"].float(), torch.ones(len(data["H_val"]), 1)), dim=1)
    test_design = torch.cat((data["H_test"].float(), torch.ones(len(data["H_test"]), 1)), dim=1)
    val_scores = val_design @ weights
    threshold, validation_ba = find_best_threshold(data["y_val"].numpy(), val_scores.numpy())
    test_scores = test_design @ weights
    metrics = calculate_metrics(data["y_test"].numpy(), test_scores.numpy(), threshold)
    return {
        "method": "Signed Target LS", "target_encoding": "stroke 0 -> -1, stroke 1 -> +1",
        "training_seed": data["training_seed"], "representation_seed": data["training_seed"],
        "split_seed": data["split_seed"], "dataset_sha256": data["dataset_sha256"],
        "preprocessing_version": data["preprocessing_version"], "balancing_method": data["balancing_method"],
        "split_class_counts": {key: data["data_metadata"].get(key) for key in (
            "original_train_class_counts", "resampled_train_class_counts", "validation_class_counts", "test_class_counts")},
        "threshold": float(threshold), "validation_balanced_accuracy": float(validation_ba),
        "weights_with_intercept": weights, **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", default=str(DATA_DIR / "stroke_dfa_representation_seed_42.pt"))
    parser.add_argument("--output", default=str(DATA_DIR / "stroke_dfa_signed_ls_results.pt"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace existing result {output}; pass --overwrite")
    result = train_signed_least_squares(args.representation)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(f"Validation BA: {result['validation_balanced_accuracy']:.4f}")
    metric_names = ("balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp")
    print("Test metrics:", {key: result[key] for key in metric_names})
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
