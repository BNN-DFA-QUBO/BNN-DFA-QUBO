import sys
from pathlib import Path
import csv

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

from utils.stroke_config import SEEDS
from utils.stroke_representation import save_representation

from experiments.stroke_experiments.stroke_bnn_dfa import train_dfa
from experiments.stroke_experiments.stroke_real_ls import train_real_ls
from experiments.stroke_experiments.stroke_dfa_binarized_ls import train_binarized_ls
from experiments.stroke_experiments.stroke_dfa_direct_binary import train_direct_binary
from experiments.stroke_experiments.stroke_dfa_ste_binary import train_ste_binary
from experiments.stroke_experiments.stroke_dfa_qubo import train_qubo


RESULTS_PATH = Path(
    "data/stroke_controlled_results.csv"
)


def run_experiments():
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    print("=" * 70)
    print("STROKE CONTROLLED EXPERIMENT")
    print("=" * 70)

    print(
        f"Seeds: {SEEDS}"
    )

    for seed in SEEDS:

        print()
        print("=" * 70)
        print(
            f"SEED {seed}"
        )
        print("=" * 70)

        representation_path = (
            f"data/stroke_dfa_representation_seed_{seed}.pt"
        )

        print()
        print("Training DFA representation...")

        dfa_result = train_dfa(seed)

        save_representation(
            dfa_result,
            representation_path,
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

    save_results(
        all_results
    )

    print()
    print("=" * 70)
    print("CONTROLLED EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Raw results saved to: "
        f"{RESULTS_PATH}"
    )

    print(
        f"Total experiments: "
        f"{len(all_results)}"
    )


def save_results(results):
    if not results:
        return

    fieldnames = [
        "method",
        "seed",
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
        RESULTS_PATH,
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
    run_experiments()