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

from experiments.stroke_experiments.stroke_bnn_dfa import train_dfa
from experiments.stroke_experiments.stroke_real_ls import train_real_ls
from experiments.stroke_experiments.stroke_dfa_binarized_ls import train_binarized_ls
from experiments.stroke_experiments.stroke_dfa_direct_binary import train_direct_binary
from experiments.stroke_experiments.stroke_dfa_ste_binary import train_ste_binary
from experiments.stroke_experiments.stroke_dfa_qubo import train_qubo


RESULTS_PATH = DATA_DIR.parent / "stroke_controlled_results.csv"


def run_experiments(seeds=None, split_seed=SPLIT_SEED, balancing_method=BALANCING_METHOD, overwrite=False,
                    output_dir=DATA_DIR):
    output_dir = Path(output_dir)
    results_path = output_dir.parent / "stroke_controlled_results.csv"
    seeds = list(SEEDS if seeds is None else seeds)
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

        save_representation(
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
            (
                "Real LS",
                train_real_ls,
                False
            ),
            (
                "Binarized LS",
                train_binarized_ls,
                False
            ),
            (
                "Direct Binary",
                train_direct_binary,
                True
            ),
            (
                "STE Binary",
                train_ste_binary,
                True
            ),
            (
                "QUBO Binary",
                train_qubo,
                True
            )
        ]

        for method_name, method_function, uses_seed in methods:

            print()
            print(
                f"Running {method_name}..."
            )

            if uses_seed:
                result = method_function(
                    representation_path,
                    seed
                )
            else:
                result = method_function(
                    representation_path
                )

            result["method"] = method_name
            result["seed"] = seed
            result["model_seed"] = seed
            result["training_seed"] = seed
            result["representation_seed"] = dfa_result["training_seed"]
            result["split_seed"] = preprocessing_metadata["split_seed"]
            result["preprocessing_version"] = preprocessing_metadata["preprocessing_version"]
            result["preprocessing_config_hash"] = preprocessing_metadata["preprocessing_config_hash"]
            result["dataset_sha256"] = preprocessing_metadata["dataset_sha256"]
            result["balancing_method"] = preprocessing_metadata["balancing_method"]
            result["head_seed"] = result.get("head_seed", "")
            result["loss_configuration"] = {
                "Real LS": "least squares with {0,1} targets and intercept",
                "Binarized LS": "least squares with {0,1} targets; binarize coefficients",
                "Direct Binary": "mean-difference direction; validation threshold",
                "STE Binary": "unweighted BCEWithLogitsLoss on balanced representations",
                "QUBO Binary": "QUBO squared error with {-1,+1} targets",
            }[method_name]
            result["split_class_counts"] = json.dumps({
                "train_original": preprocessing_metadata["original_train_class_counts"],
                "train_resampled": preprocessing_metadata["resampled_train_class_counts"],
                "validation": preprocessing_metadata["validation_class_counts"],
                "test": preprocessing_metadata["test_class_counts"],
            }, sort_keys=True)
            result["representation_path"] = (
                representation_path
            )

            all_results.append(
                result
            )

            print(
                f"Test BA: "
                f"{result['balanced_accuracy'] * 100:.4f}%"
            )

            print(
                f"F1: "
                f"{result['f1'] * 100:.4f}%"
            )

            print(
                f"ROC-AUC: "
                f"{result['roc_auc'] * 100:.4f}%"
            )

            print(
                f"PR-AUC: "
                f"{result['pr_auc'] * 100:.4f}%"
            )

    save_results(all_results, results_path)

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


def save_results(results, results_path=None):
    if not results:
        return
    results_path = RESULTS_PATH if results_path is None else Path(results_path)

    fieldnames = [
        "method",
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
        "alpha",
        "bias",
        "qubo_num_reads",
        "head_seed",
        "representation_path"
    ]

    with open(
        results_path,
        "w",
        newline=""
    ) as file:

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
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs and regenerate incompatible preprocessing artifacts")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR,
                        help="Directory for preprocessing and representation artifacts (default: data/stroke)")
    args = parser.parse_args()
    run_experiments(seeds=args.seeds, split_seed=args.split_seed, balancing_method=args.balancing,
                    overwrite=args.overwrite, output_dir=args.output_dir)
