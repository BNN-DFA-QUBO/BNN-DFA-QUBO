import sys
from pathlib import Path
import csv
import argparse
import json

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

from utils.stroke_config import BALANCING_METHOD, SEEDS, SPLIT_SEED
from utils.stroke_representation import save_representation
from utils.stroke_data import DATA_DIR, load_stroke_metadata, prepare_stroke_data
from utils.run_provenance import collect_run_provenance, stable_run_id

from experiments.stroke_experiments.stroke_bnn_dfa import train_dfa
from experiments.stroke_experiments.stroke_real_ls import train_real_ls
from experiments.stroke_experiments.stroke_dfa_binarized_ls import train_binarized_ls
from experiments.stroke_experiments.stroke_dfa_direct_binary import train_direct_binary
from experiments.stroke_experiments.stroke_dfa_ste_binary import train_ste_binary
from experiments.stroke_experiments.stroke_dfa_qubo import train_qubo


RESULTS_PATH = DATA_DIR.parent / "stroke_controlled_results.csv"

# These describe the established controlled-comparison implementations; they do
# not imply that every head optimizes the same objective.
CONTROLLED_METHOD_METADATA = {
    "Real LS": {
        "method_variant": "continuous_least_squares",
        "objective": "squared residual error",
        "target_encoding": "stroke labels {0,1}",
        "bias_treatment": "intercept fitted jointly with weights by least squares",
        "optimization_procedure": "numpy.linalg.lstsq closed-form fit; validation-selected threshold",
    },
    "Binarized LS": {
        "method_variant": "least_squares_fit_then_binary_weights",
        "objective": "squared residual error before weight sign binarization",
        "target_encoding": "stroke labels {0,1}",
        "bias_treatment": "continuous least-squares intercept retained",
        "optimization_procedure": "numpy.linalg.lstsq, sign-binarize weights, validation-selected threshold",
    },
    "Binarized Ridge LS (refit scale+bias)": {
        "method_variant": "standardized_ridge_ls_then_binary_weights_refit_scale_bias",
        "objective": "regularized squared residual error for initial weights; training-only least-squares refit after binarization",
        "target_encoding": "stroke labels {0,1}",
        "bias_treatment": "class-independent scalar scale and intercept refitted on training rows after sign binarization",
        "optimization_procedure": "train-only hidden standardization; float64 augmented SVD Ridge LS; sign-binarize feature weights; training-only scale/intercept refit; validation-selected threshold",
    },
    "Direct Binary": {
        "method_variant": "standardized_class_mean_sign_direction",
        "objective": "no loss optimization; sign of positive-minus-negative training mean",
        "target_encoding": "stroke labels {0,1} define the two class means",
        "bias_treatment": "fixed at zero",
        "optimization_procedure": "train-only hidden standardization; analytic class means; sign of class-mean difference; validation-selected threshold",
    },
    "STE Binary": {
        "method_variant": "ste_bce_standardized_validation_selected_initialization",
        "objective": "binary cross entropy with logits",
        "target_encoding": "stroke labels {0,1}",
        "bias_treatment": "learned scalar bias",
        "optimization_procedure": "train-only standardization of hidden units; three training-only initializations (Ridge direction, class-mean direction, random direction); confidence-weighted latent STE starts place low-confidence signs near zero; Adam STE with logits divided by sqrt(hidden_size); evaluate epoch-zero baseline and retain best validation checkpoint with patience-based stopping; record actual weight sign flips; select initialization/checkpoint by equal-weight validation BA/F1/PR-AUC/ROC-AUC composite; ties favor balanced accuracy; validation-selected threshold",
    },
    "QUBO Binary": {
        "method_variant": "qubo_signed_target_squared_error_validation_scale_sweep",
        "objective": "QUBO squared error on signed targets {-1,+1}",
        "target_encoding": "stroke 0 -> -1; stroke 1 -> +1",
        "bias_treatment": "recovered as the training residual offset after train-only hidden standardization",
        "optimization_procedure": "train-only standardization; simulated annealing across configured alpha multipliers with configured sweeps per read; low-energy samples plus class-mean, Ridge, and random training-only starts refined under the training-derived Ising energy; candidate selection by equal-weight validation BA/F1/PR-AUC/ROC-AUC composite; ties favor balanced accuracy; validation-selected threshold",
    },
}


def run_experiments(seeds=None, split_seed=SPLIT_SEED, balancing_method=BALANCING_METHOD, overwrite=False,
                    output_dir=DATA_DIR, ls_solver="ridge"):
    output_dir = Path(output_dir)
    results_path = output_dir.parent / "stroke_controlled_results.csv"
    if ls_solver not in {"legacy", "ridge", "both"}:
        raise ValueError("ls_solver must be 'legacy', 'ridge', or 'both'")
    if ls_solver == "legacy":
        binary_methods = [("Binarized LS", "legacy")]
    elif ls_solver == "ridge":
        binary_methods = [("Binarized Ridge LS (refit scale+bias)", "ridge")]
    else:
        binary_methods = [
            ("Binarized LS", "legacy"),
            ("Binarized Ridge LS (refit scale+bias)", "ridge"),
        ]
    seeds = list(SEEDS if seeds is None else seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate model seeds would create duplicate controlled-run records")
    run_provenance = collect_run_provenance()
    representation_paths = [output_dir / f"stroke_dfa_representation_seed_{seed}.pt" for seed in seeds]
    model_result_paths = [output_dir / f"stroke_bnn_dfa_seed_{seed}.json" for seed in seeds]
    existing_outputs = [str(path) for path in representation_paths + model_result_paths if path.exists()]
    if results_path.exists():
        existing_outputs.append(str(results_path))
    if existing_outputs and not overwrite:
        raise FileExistsError("Existing controlled-run outputs would be replaced: " + ", ".join(existing_outputs)
                              + ". Pass --overwrite to replace them.")
    # Fix the shared split and preprocessing before any model seed starts.
    prepare_stroke_data(split_seed=split_seed, balance_method=balancing_method, overwrite=overwrite,
                        output_dir=output_dir)
    preprocessing_metadata = load_stroke_metadata(output_dir / "metadata.json")
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    print("=" * 70)
    print("STROKE CONTROLLED EXPERIMENT")
    print("=" * 70)

    print(
        f"Model training seeds: {seeds}; split seed: {split_seed}"
    )

    for seed in seeds:

        print()
        print("=" * 70)
        print(
            f"SEED {seed}"
        )
        print("=" * 70)

        representation_path = output_dir / f"stroke_dfa_representation_seed_{seed}.pt"

        print()
        print("Training DFA representation...")

        dfa_result = train_dfa(seed, split_seed=split_seed, balance_method=balancing_method,
                               overwrite=overwrite, output_dir=output_dir)

        representation_artifact = save_representation(
            dfa_result,
            str(representation_path),
            feature_names=dfa_result["feature_names"]
        )

        print(
            f"DFA validation BA: "
            f"{dfa_result['best_val_ba'] * 100:.4f}%"
        )

        print(
            f"DFA best epoch: "
            f"{dfa_result['best_epoch']}"
        )

        methods = [
            ("Real LS", train_real_ls, "single"),
            *[
                (method_name, train_binarized_ls, solver)
                for method_name, solver in binary_methods
            ],
            ("Direct Binary", train_direct_binary, "seed"),
            ("STE Binary", train_ste_binary, "seed"),
            ("QUBO Binary", train_qubo, "seed"),
        ]

        for method_name, method_function, method_mode in methods:

            print()
            print(f"Running {method_name}...")

            failed = False
            import time
            t_fit_start = time.perf_counter()
            try:
                if method_mode == "seed":
                    result = method_function(representation_path, seed)
                elif method_mode in ("legacy", "ridge"):
                    result = method_function(representation_path, ls_solver=method_mode)
                else:
                    result = method_function(representation_path)
                head_fit_time = time.perf_counter() - t_fit_start
            except ValueError as exc:
                head_fit_time = time.perf_counter() - t_fit_start
                # A degenerate QUBO scale is a scientific failure for this head,
                # but should not erase successful results from the other seeds.
                if method_name != "QUBO Binary" or "QUBO scale calibration is degenerate" not in str(exc):
                    raise
                result = {
                    "method": method_name,
                    "status": "failed",
                    "failure_reason": str(exc),
                }
                failed = True

            result["method"] = method_name
            result.setdefault("status", "ok")
            result.setdefault("failure_reason", "")
            result["experiment_type"] = "controlled_end_to_end_head_comparison"
            result["seed"] = seed
            result["model_seed"] = seed
            result["training_seed"] = seed
            result["representation_seed"] = dfa_result["training_seed"]
            result["split_seed"] = preprocessing_metadata["split_seed"]
            result["preprocessing_version"] = preprocessing_metadata["preprocessing_version"]
            result["preprocessing_config_hash"] = preprocessing_metadata["preprocessing_config_hash"]
            result["dataset_sha256"] = preprocessing_metadata["dataset_sha256"]
            result["balancing_method"] = preprocessing_metadata["balancing_method"]
            result["head_seed"] = result.get("head_seed", seed if method_mode == "seed" else "")
            result.update(CONTROLLED_METHOD_METADATA[method_name])
            result["loss_configuration"] = CONTROLLED_METHOD_METADATA[method_name]["objective"]
            result["split_class_counts"] = json.dumps({
                "train_original": preprocessing_metadata["original_train_class_counts"],
                "train_resampled": preprocessing_metadata["resampled_train_class_counts"],
                "validation": preprocessing_metadata["validation_class_counts"],
                "test": preprocessing_metadata["test_class_counts"],
            }, sort_keys=True)
            result["representation_path"] = str(representation_path)
            result["representation_sha256"] = representation_artifact["representation_sha256"]
            result["run_id"] = stable_run_id("stroke_controlled_end_to_end", {
                "method": method_name, "seed": seed,
                "split_seed": preprocessing_metadata["split_seed"],
                "dataset_sha256": preprocessing_metadata["dataset_sha256"],
                "preprocessing_config_hash": preprocessing_metadata["preprocessing_config_hash"],
                "representation_sha256": representation_artifact["representation_sha256"],
                "method_variant": result["method_variant"],
                "alpha": result.get("alpha"),
                "alpha_multiplier": result.get("alpha_multiplier"),
                "qubo_num_reads": result.get("qubo_num_reads"),
                "qubo_num_sweeps": result.get("qubo_num_sweeps"),
                "qubo_solver_seed_selected": result.get("qubo_solver_seed_selected"),
            }, run_provenance)
            result["git_commit_sha"] = run_provenance.get("git_commit_sha", "unavailable")
            result["git_worktree_dirty"] = run_provenance.get("git_worktree_dirty", "unavailable")
            result["git_worktree_fingerprint"] = run_provenance.get("git_worktree_fingerprint", "unavailable")
            result["python_version"] = run_provenance.get("python_version", "unavailable")
            result["os_platform"] = run_provenance.get("os_platform", "unavailable")
            result["package_versions_json"] = json.dumps(run_provenance.get("package_versions", {}), sort_keys=True)
            result["provenance_json"] = json.dumps(run_provenance, sort_keys=True)
            result["representation_provenance_json"] = json.dumps(
                representation_artifact.get("run_provenance", {}), sort_keys=True
            )

            # Computational-cost accounting:
            # Backbone: input_size=15 -> 15*64 + 64 = 1024; 64*1 + 1 = 65 -> 1089
            backbone_param_count = 15 * 64 + 64 + 64 * 1 + 1
            if method_name in ("Real LS", "Binarized LS"):
                head_param_count = 65  # 64 weights + 1 bias
            elif method_name == "Binarized Ridge LS (refit scale+bias)":
                head_param_count = 66  # 64 binary weights + 1 scale + 1 bias
            elif method_name == "Direct Binary":
                head_param_count = 64  # 64 binary weights (bias fixed at 0)
            elif method_name == "STE Binary":
                head_param_count = 65  # 64 deployed binary weights + 1 bias
            elif method_name == "QUBO Binary":
                head_param_count = 66  # 64 binary weights + 1 scale + 1 bias
            else:
                head_param_count = 65
            total_param_count = backbone_param_count + head_param_count

            # Timing accounting
            # For QUBO, qubo_search_seconds is already measured accurately inside train_qubo
            if method_name == "QUBO Binary" and "qubo_search_seconds" in result:
                actual_head_fit = float(result["qubo_search_seconds"])
            else:
                actual_head_fit = float(head_fit_time)

            cur_backbone_time = float(dfa_result.get("training_seconds", 0.0))
            cur_extract_time = float(dfa_result.get("feature_extraction_seconds", 0.0))
            cur_run_total = cur_backbone_time + cur_extract_time + actual_head_fit

            result["head_fit_time_sec"] = actual_head_fit
            result["backbone_training_time_sec"] = cur_backbone_time
            result["feature_extraction_time_sec"] = cur_extract_time
            result["current_run_total_time_sec"] = cur_run_total
            result["end_to_end_training_time_sec"] = cur_run_total
            result["pipeline_timing_status"] = "complete"
            result["backbone_parameter_count"] = backbone_param_count
            result["head_parameter_count"] = head_param_count
            result["total_parameter_count"] = total_param_count

            all_results.append(result)

            if failed:
                print(f"QUBO head FAILED for seed {seed}: {result['failure_reason']}")
            else:
                print(f"Test BA: {result['balanced_accuracy'] * 100:.4f}%")
                print(f"F1: {result['f1'] * 100:.4f}%")
                print(f"ROC-AUC: {result['roc_auc'] * 100:.4f}%")
                print(f"PR-AUC: {result['pr_auc'] * 100:.4f}%")

    save_results(all_results, results_path, overwrite=overwrite)

    print()
    print("=" * 70)
    print("CONTROLLED EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Raw results saved to: "
        f"{results_path}"
    )

    print(
        f"Total experiments: "
        f"{len(all_results)}"
    )


def save_results(results, results_path=None, overwrite=False):
    if not results:
        return
    results_path = RESULTS_PATH if results_path is None else Path(results_path)

    fieldnames = [
        "experiment_type",
        "method",
        "status",
        "failure_reason",
        "method_variant",
        "seed",
        "model_seed",
        "training_seed",
        "split_seed",
        "representation_seed",
        "preprocessing_version",
        "preprocessing_config_hash",
        "dataset_sha256",
        "balancing_method",
        "loss_configuration",
        "objective",
        "target_encoding",
        "bias_treatment",
        "optimization_procedure",
        "split_class_counts",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "predicted_positive",
        "actual_positive",
        "threshold",
        "validation_balanced_accuracy",
        "validation_selection_score",
        "validation_f1",
        "validation_pr_auc",
        "validation_roc_auc",
        "alpha",
        "base_alpha",
        "alpha_multiplier",
        "qubo_alpha_multipliers",
        "qubo_candidate_count",
        "qubo_top_candidates_per_scale",
        "qubo_search_seconds",
        "qubo_candidate_generation_seconds",
        "qubo_validation_selection_seconds",
        "qubo_energy",
        "qubo_candidate_source",
        "bias",
        "qubo_num_reads",
        "qubo_num_reads_per_scale",
        "qubo_num_sweeps",
        "qubo_solver_seed_base",
        "qubo_solver_seed_selected",
        "qubo_solver_seed_rule",
        "qubo_random_seed_starts",
        "qubo_total_reads",
        "initialization",
        "candidate_initializations",
        "ste_epochs",
        "ste_epochs_run",
        "ste_best_epoch",
        "ste_patience",
        "ste_validation_checkpoints",
        "ste_weight_flips_from_init",
        "ste_final_weight_flips_from_init",
        "ste_latent_init",
        "ste_learning_rate",
        "training_loss",
        "hidden_feature_transform",
        "standardized_constant_units",
        "head_seed",
        "representation_path",
        "representation_sha256",
        "run_id",
        "git_commit_sha",
        "git_worktree_dirty",
        "git_worktree_fingerprint",
        "python_version",
        "os_platform",
        "package_versions_json",
        "provenance_json",
        "representation_provenance_json",
        "backbone_training_time_sec",
        "feature_extraction_time_sec",
        "head_fit_time_sec",
        "current_run_total_time_sec",
        "end_to_end_training_time_sec",
        "pipeline_timing_status",
        "backbone_parameter_count",
        "head_parameter_count",
        "total_parameter_count",
    ]

    if results_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing controlled results: {results_path}")
    run_ids = [row.get("run_id") for row in results if row.get("run_id")]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate run_id values found in controlled results")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w" if overwrite else "x", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        for result in results:

            row = {
                field: result.get(
                    field,
                    ""
                )
                for field in fieldnames
            }

            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stroke DFA representation and controlled head comparisons")
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="BNN/DFA model-training seeds")
    parser.add_argument("--balancing", choices=("random_oversample", "none"), default=BALANCING_METHOD)
    parser.add_argument("--ls-solver", choices=("legacy", "ridge", "both"), default="ridge",
                        help="Stroke Binarized-LS implementation(s); default is Ridge; both evaluates legacy and Ridge variants on each same representation")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs and regenerate incompatible preprocessing artifacts")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR,
                        help="Directory for preprocessing and representation artifacts (default: data/stroke)")
    args = parser.parse_args()
    print(f"Stroke Binarized-LS solver: {args.ls_solver}")
    run_experiments(seeds=args.seeds, split_seed=args.split_seed, balancing_method=args.balancing,
                    overwrite=args.overwrite, output_dir=args.output_dir, ls_solver=args.ls_solver)
