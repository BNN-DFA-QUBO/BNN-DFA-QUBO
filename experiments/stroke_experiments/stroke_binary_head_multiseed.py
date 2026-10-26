import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

import random
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from utils.stroke_representation import load_stroke_head_representation
from utils.stroke_binary_features import standardize_hidden_features
from utils.stroke_data import DATA_DIR
from utils.stroke_metrics import find_best_threshold as shared_find_best_threshold
from utils.run_provenance import collect_run_provenance, stable_run_id



# These are head initialization/optimization seeds on one fixed representation.
SEEDS = [42, 123, 2024, 7, 99]
NUM_READS = 100
STE_EPOCHS = 200
DIRECT_MAX_PASSES = 20

MULTISEED_METHOD_METADATA = {
    "Direct Binary": {
        "method_variant": "coordinate_flip_bce_direct_binary",
        "objective": "binary_cross_entropy_with_logits",
        "target_encoding": "stroke labels {0,1}",
        "bias_treatment": "fixed at zero",
        "optimization_procedure": "train-only hidden standardization; coordinate-wise sign flips from all-positive weights, greedy BCE improvement, at most 20 passes; validation-selected threshold",
        "loss_configuration": "unweighted BCEWithLogitsLoss-style objective on fixed-representation training rows",
    },
    "STE Binary": {
        "method_variant": "ste_bce_standardized_validation_selected_initialization_fixed_representation",
        "objective": "binary cross entropy with logits",
        "target_encoding": "stroke labels {0,1}",
        "bias_treatment": "learned scalar bias",
        "optimization_procedure": "shared controlled STE head; train-only hidden standardization; three initializations selected by equal-weight validation BA/F1/PR-AUC/ROC-AUC composite",
        "loss_configuration": "BCEWithLogitsLoss on train-standardized fixed-representation features; logits divided by sqrt(hidden_size)",
    },
    "QUBO Binary": {
        "method_variant": "qubo_signed_target_squared_error_alpha_sweep_fixed_representation",
        "objective": "QUBO squared error on signed targets {-1,+1}",
        "target_encoding": "stroke 0 -> -1; stroke 1 -> +1",
        "bias_treatment": "recovered as the training residual offset after train-only hidden standardization",
        "optimization_procedure": "shared controlled QUBO head; alpha-multiplier sweep, low-energy candidates selected by equal-weight validation BA/F1/PR-AUC/ROC-AUC composite",
        "loss_configuration": "QUBO squared-error objective on train-standardized fixed-representation features",
    },
}


def qubo_result_columns(result):
    """Keep the selected QUBO configuration and timer fields in raw records."""
    names = (
        "alpha", "base_alpha", "alpha_multiplier", "qubo_num_reads",
        "qubo_num_reads_per_scale", "qubo_num_sweeps", "qubo_total_reads",
        "qubo_random_seed_starts", "qubo_top_candidates_per_scale",
        "qubo_solver_seed_base", "qubo_solver_seed_selected", "qubo_solver_seed_rule",
        "qubo_candidate_count", "qubo_energy", "qubo_candidate_source",
        "qubo_search_seconds", "qubo_candidate_generation_seconds",
        "qubo_validation_selection_seconds", "validation_balanced_accuracy",
        "validation_selection_score", "validation_f1", "validation_pr_auc",
        "validation_roc_auc", "threshold", "bias", "hidden_feature_transform",
        "standardized_constant_units",
    )
    output = {name: result[name] for name in names if name in result}
    if "qubo_alpha_multipliers" in result:
        output["qubo_alpha_multipliers"] = json.dumps(result["qubo_alpha_multipliers"])
    return output


def write_multiseed_results(output_path, summary_path, rows, overwrite=False):
    """Write fixed-representation raw/summary CSVs with duplicate-run checks."""
    output_path, summary_path = Path(output_path), Path(summary_path)
    existing = [str(path) for path in (output_path, summary_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to replace existing multi-seed outputs: " + ", ".join(existing))
    run_ids = [row.get("run_id") for row in rows if row.get("run_id")]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate run_id values found in fixed-representation results")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    group_keys = [key for key in (
        "method", "experiment_type", "dataset_sha256", "preprocessing_version",
        "preprocessing_config_hash", "split_seed", "representation_seed", "balancing_method",
        "qubo_alpha_multipliers", "qubo_num_reads_per_scale", "qubo_num_sweeps",
        "qubo_solver_seed_rule", "representation_sha256",
    ) if key in df.columns]
    metric_fields = [key for key in (
        "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    ) if key in df.columns]
    summary = df.groupby(group_keys, dropna=False)[metric_fields].agg(["mean", "std"])
    summary.to_csv(summary_path)
    return output_path, summary_path, summary


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(representation_path):
    representation = load_stroke_head_representation(representation_path)

    H_train, H_val, H_test, transform = standardize_hidden_features(
        representation["H_train"], representation["H_val"], representation["H_test"]
    )
    representation["head_hidden_feature_transform"] = transform["transform"]
    representation["head_standardized_constant_units"] = transform["constant_units"]

    # Labels are stored with the representations so a separately regenerated split
    # can never silently misalign examples and learned features.
    y_train = representation["y_train"].float()
    y_val = representation["y_val"].long()
    y_test = representation["y_test"].long()
    if not (len(H_train) == len(y_train) and len(H_val) == len(y_val) and len(H_test) == len(y_test)):
        raise ValueError("Representation rows and embedded labels are misaligned")

    return (
        H_train,
        H_val,
        H_test,
        y_train,
        y_val,
        y_test,
        representation
    )


def find_best_threshold(y_true, scores):
    threshold, _ = shared_find_best_threshold(y_true.numpy(), scores.numpy())
    return float(threshold)


def evaluate(
    y_true,
    scores,
    threshold
):
    predictions = (
        scores >= threshold
    ).long()

    y = y_true.numpy()
    p = predictions.numpy()
    s = scores.numpy()
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()

    return {
        "balanced_accuracy":
            balanced_accuracy_score(y, p),

        "precision":
            precision_score(
                y,
                p,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                p,
                zero_division=0
            ),

        "f1":
            f1_score(
                y,
                p,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(y, s),

        "pr_auc":
            average_precision_score(y, s),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold)
    }


# ============================================================
# DIRECT BINARY
# ============================================================

def direct_binary(
    H,
    labels,
    seed
):
    set_seed(seed)

    weights = torch.ones(
        H.shape[1],
        dtype=torch.float32
    )

    target = labels.clone()

    best_loss = float("inf")

    for _ in range(DIRECT_MAX_PASSES):

        improved = False

        for i in torch.randperm(
            H.shape[1]
        ).tolist():

            candidate = weights.clone()

            candidate[i] *= -1

            scores = H @ candidate

            loss = nn.functional.binary_cross_entropy_with_logits(
                scores,
                target
            )

            if loss.item() < best_loss:

                best_loss = loss.item()
                weights = candidate

                improved = True

        if not improved:
            break

    return weights


# STE and QUBO are delegated to the shared controlled implementations inside
# main(), so the diagnostic CLI cannot silently use the older unstable copies.

# ============================================================
# MAIN
# ============================================================

def main(representation_path=DATA_DIR / "stroke_dfa_representation_seed_42.pt", seeds=None,
         output_path=DATA_DIR / "stroke_binary_head_multiseed.csv", overwrite=False):
    summary_path = str(Path(output_path).with_name(Path(output_path).stem + "_summary.csv"))
    existing = [path for path in (output_path, summary_path) if Path(path).exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to replace existing multi-seed outputs: " + ", ".join(existing) + "; pass --overwrite")
    seeds = list(SEEDS if seeds is None else seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate head seeds would create duplicate fixed-representation records")

    (
        H_train,
        H_val,
        H_test,
        y_train,
        y_val,
        y_test,
        representation
    ) = load_data(representation_path)
    provenance = representation["data_metadata"]
    representation_provenance = representation.get("run_provenance", {})
    run_provenance = collect_run_provenance()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("H_train:", H_train.shape)
    print("H_val:", H_val.shape)
    print("H_test:", H_test.shape)

    results = []

    for seed in seeds:

        print()
        print("=" * 60)
        print(f"SEED {seed}")
        print("=" * 60)

        # ----------------------------------------------------
        # Direct Binary
        # ----------------------------------------------------

        weights = direct_binary(
            H_train,
            y_train,
            seed
        )

        train_scores = H_train @ weights
        val_scores = H_val @ weights
        test_scores = H_test @ weights

        threshold = find_best_threshold(
            y_val,
            val_scores
        )

        metrics = evaluate(
            y_test,
            test_scores,
            threshold
        )

        results.append(
            {
                "seed": seed,
                "head_seed": seed,
                "model_seed": representation["training_seed"],
                "training_seed": representation["training_seed"],
                "representation_seed": representation["training_seed"],
                "split_seed": representation["split_seed"],
                "dataset_sha256": representation["dataset_sha256"],
                "preprocessing_version": representation["preprocessing_version"],
                "preprocessing_config_hash": provenance.get("preprocessing_config_hash"),
                "balancing_method": representation["balancing_method"],
                "experiment_type": "fixed_representation_head_seed_variability",
                "representation_path": str(representation_path),
                "representation_sha256": representation.get("representation_sha256", ""),
                "original_train_class_counts": json.dumps(provenance["original_train_class_counts"], sort_keys=True),
                "resampled_train_class_counts": json.dumps(provenance["resampled_train_class_counts"], sort_keys=True),
                "validation_class_counts": json.dumps(provenance["validation_class_counts"], sort_keys=True),
                "test_class_counts": json.dumps(provenance["test_class_counts"], sort_keys=True),
                "method": "Direct Binary",
                **MULTISEED_METHOD_METADATA["Direct Binary"],
                "bias": 0.0,
                "hidden_feature_transform": representation["head_hidden_feature_transform"],
                "standardized_constant_units": representation["head_standardized_constant_units"],
                **metrics
            }
        )

        print(
            f"Direct Binary: "
            f"BA={metrics['balanced_accuracy'] * 100:.4f}%"
        )

        # ----------------------------------------------------
        # STE Binary (shared stable implementation)
        # ----------------------------------------------------

        from experiments.stroke_experiments.stroke_dfa_ste_binary import train_ste_binary
        ste_result = train_ste_binary(representation_path, seed)
        ste_metric_fields = (
            "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
            "tn", "fp", "fn", "tp", "predicted_positive", "actual_positive", "threshold",
        )
        results.append(
            {
                "seed": seed,
                "head_seed": seed,
                "model_seed": representation["training_seed"],
                "training_seed": representation["training_seed"],
                "representation_seed": representation["training_seed"],
                "split_seed": representation["split_seed"],
                "dataset_sha256": representation["dataset_sha256"],
                "preprocessing_version": representation["preprocessing_version"],
                "preprocessing_config_hash": provenance.get("preprocessing_config_hash"),
                "balancing_method": representation["balancing_method"],
                "experiment_type": "fixed_representation_head_seed_variability",
                "representation_path": str(representation_path),
                "representation_sha256": representation.get("representation_sha256", ""),
                "original_train_class_counts": json.dumps(provenance["original_train_class_counts"], sort_keys=True),
                "resampled_train_class_counts": json.dumps(provenance["resampled_train_class_counts"], sort_keys=True),
                "validation_class_counts": json.dumps(provenance["validation_class_counts"], sort_keys=True),
                "test_class_counts": json.dumps(provenance["test_class_counts"], sort_keys=True),
                "method": "STE Binary",
                **MULTISEED_METHOD_METADATA["STE Binary"],
                "bias": ste_result["bias"],
                "initialization": ste_result["initialization"],
                "candidate_initializations": ste_result["candidate_initializations"],
                "validation_balanced_accuracy": ste_result["validation_balanced_accuracy"],
                "validation_selection_score": ste_result["validation_selection_score"],
                "validation_f1": ste_result["validation_f1"],
                "validation_pr_auc": ste_result["validation_pr_auc"],
                "validation_roc_auc": ste_result["validation_roc_auc"],
                "hidden_feature_transform": ste_result["hidden_feature_transform"],
                "standardized_constant_units": ste_result["standardized_constant_units"],
                "ste_epochs": ste_result["ste_epochs"],
                "ste_learning_rate": ste_result["ste_learning_rate"],
                **{key: ste_result[key] for key in ste_metric_fields},
            }
        )
        print(
            f"STE Binary: BA={ste_result['balanced_accuracy'] * 100:.4f}% "
            f"(validation init={ste_result['initialization']})"
        )

        # ----------------------------------------------------
        # QUBO Binary (shared scale-search implementation)
        # ----------------------------------------------------

        from experiments.stroke_experiments.stroke_dfa_qubo import train_qubo
        qubo_result = train_qubo(representation_path, seed)
        qubo_metric_fields = (
            "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
            "tn", "fp", "fn", "tp", "predicted_positive", "actual_positive", "threshold",
        )
        results.append(
            {
                "seed": seed,
                "head_seed": seed,
                "model_seed": representation["training_seed"],
                "training_seed": representation["training_seed"],
                "representation_seed": representation["training_seed"],
                "split_seed": representation["split_seed"],
                "dataset_sha256": representation["dataset_sha256"],
                "preprocessing_version": representation["preprocessing_version"],
                "preprocessing_config_hash": provenance.get("preprocessing_config_hash"),
                "balancing_method": representation["balancing_method"],
                "experiment_type": "fixed_representation_head_seed_variability",
                "representation_path": str(representation_path),
                "representation_sha256": representation.get("representation_sha256", ""),
                "original_train_class_counts": json.dumps(provenance["original_train_class_counts"], sort_keys=True),
                "resampled_train_class_counts": json.dumps(provenance["resampled_train_class_counts"], sort_keys=True),
                "validation_class_counts": json.dumps(provenance["validation_class_counts"], sort_keys=True),
                "test_class_counts": json.dumps(provenance["test_class_counts"], sort_keys=True),
                "method": "QUBO Binary",
                **MULTISEED_METHOD_METADATA["QUBO Binary"],
                **qubo_result_columns(qubo_result),
                **{key: qubo_result[key] for key in qubo_metric_fields},
            }
        )
        print(
            f"QUBO Binary: BA={qubo_result['balanced_accuracy'] * 100:.4f}% "
            f"(alpha multiplier={qubo_result['alpha_multiplier']}, "
            f"candidates={qubo_result['qubo_candidate_count']})"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    for row in results:
        stable_config = {
            key: row.get(key) for key in (
                "method_variant", "training_seed", "head_seed", "split_seed",
                "dataset_sha256", "preprocessing_config_hash", "balancing_method",
                "representation_sha256",
                "alpha", "base_alpha", "alpha_multiplier", "qubo_alpha_multipliers",
                "qubo_num_reads_per_scale", "qubo_num_sweeps", "qubo_solver_seed_selected",
            ) if key in row
        }
        row["run_id"] = stable_run_id(row["experiment_type"], stable_config, run_provenance)
        row["git_commit_sha"] = run_provenance.get("git_commit_sha", "unavailable")
        row["git_worktree_dirty"] = run_provenance.get("git_worktree_dirty", "unavailable")
        row["python_version"] = run_provenance.get("python_version", "unavailable")
        row["os_platform"] = run_provenance.get("os_platform", "unavailable")
        row["package_versions_json"] = json.dumps(run_provenance.get("package_versions", {}), sort_keys=True)
        row["provenance_json"] = json.dumps(run_provenance, sort_keys=True)
        row["representation_provenance_json"] = json.dumps(representation_provenance, sort_keys=True)

    df = pd.DataFrame(results)

    print()
    print("=" * 60)
    print("PER-SEED RESULTS")
    print("=" * 60)

    print(
        df.to_string(
            index=False
        )
    )

    _, written_summary_path, summary = write_multiseed_results(
        output_path, summary_path, results, overwrite=overwrite,
    )

    print()
    print("=" * 60)
    print("MEAN ± STANDARD DEVIATION")
    print("=" * 60)

    print(summary)


    print()
    print(
        "Saved:"
    )

    print(
        output_path
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure binary-head initialization variability on one fixed DFA representation")
    parser.add_argument("--representation", default=str(DATA_DIR / "stroke_dfa_representation_seed_42.pt"))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output", default=str(DATA_DIR / "stroke_binary_head_multiseed.csv"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.representation, args.seeds, args.output, args.overwrite)
