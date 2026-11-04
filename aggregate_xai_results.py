import csv
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path("data/xai_results")
OUT = Path("aggregated")
OUT.mkdir(exist_ok=True)

EXPECTED_SEEDS = {"42", "123", "2024", "7", "99"}

REQUIRED_FILES = [
    "summary_metrics.csv",
    "sample_metrics.csv",
    "feature_attributions.csv",
    "VALIDATION_REPORT.md",
    "run_manifest.json",
]

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "shap_additivity_pass_rate",
    "mean_lime_local_r2",
    "mean_shap_lime_rank_spearman",
    "mean_shap_stability_spearman",
    "mean_lime_stability_spearman",
]


def save_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------
# 1. DISCOVER AND VALIDATE RUNS
# --------------------------------------------------

groups = defaultdict(list)

for manifest_path in ROOT.glob("*/*/*/run_manifest.json"):

    run_dir = manifest_path.parent

    try:
        with open(manifest_path) as f:
            m = json.load(f)

        config = m.get("training", {})
        seed = config.get("seed")

        if seed is None:
            seed = m.get("config", {}).get("seed")

        if seed is None:
            seed = m.get("seed")

        if seed is None:
            print("SKIPPING: Missing seed:", run_dir)
            continue

        seed = str(seed)

        dataset = run_dir.parent.parent.name
        method = run_dir.parent.name

        missing = [
            filename for filename in REQUIRED_FILES
            if not (run_dir / filename).is_file()
        ]

        if missing:
            print(
                f"INVALID RUN: {dataset}/{method}/seed_{seed}"
                f" | Missing: {missing}"
            )
            continue

        groups[(dataset, method, seed)].append({
            "path": run_dir,
            "manifest": m,
            "timestamp": manifest_path.stat().st_mtime,
        })

    except Exception as e:
        print("ERROR:", manifest_path, e)


# --------------------------------------------------
# 2. SELECT LATEST VALID RUN PER COMBINATION
# --------------------------------------------------

selected_runs = []
duplicate_count = 0

for key, candidates in sorted(groups.items()):

    latest = max(
        candidates,
        key=lambda x: x["timestamp"]
    )

    duplicate_count += len(candidates) - 1

    m = latest["manifest"]

    evaluation = m.get("test_evaluation", {})
    validation = m.get("validation_summary", {})
    config = m.get("training", {})
    backbone = m.get("backbone", {})

    seed = key[2]

    row = {
        "dataset_folder": key[0],
        "method": key[1],
        "seed": seed,
        "backbone_class": backbone.get("class", ""),
        "representation_dim": m.get("representation_dim"),

        "accuracy": evaluation.get("accuracy"),
        "balanced_accuracy": evaluation.get("balanced_accuracy"),
        "macro_f1": evaluation.get("macro_f1"),

        "shap_additivity_pass_rate":
            validation.get("shap_additivity_pass_rate"),

        "mean_lime_local_r2":
            validation.get("mean_lime_local_r2"),

        "mean_shap_lime_rank_spearman":
            validation.get("mean_shap_lime_rank_spearman"),

        "mean_shap_stability_spearman":
            validation.get("mean_shap_stability_spearman"),

        "mean_lime_stability_spearman":
            validation.get("mean_lime_stability_spearman"),

        "run_directory": str(latest["path"]),
    }

    selected_runs.append(row)


# --------------------------------------------------
# 3. COMPLETENESS CHECK
# --------------------------------------------------

expected_datasets = {
    "bank",
    "mnist",
    "cifar10",
    "stroke",
}

expected_methods = {
    "real_ls",
    "binarized_ls",
    "direct_binary",
    "ste_binary",
    "qubo",
}

expected_combinations = {
    (dataset, method, seed)
    for dataset in expected_datasets
    for method in expected_methods
    for seed in EXPECTED_SEEDS
}

actual_combinations = {
    (r["dataset_folder"], r["method"], str(r["seed"]))
    for r in selected_runs
}

missing_combinations = expected_combinations - actual_combinations
unexpected_combinations = actual_combinations - expected_combinations

print("\n========== VALIDATION REPORT ==========")

print("Valid unique runs:", len(selected_runs))
print("Duplicate valid runs excluded:", duplicate_count)
print("Expected combinations:", len(expected_combinations))
print("Missing combinations:", len(missing_combinations))
print("Unexpected combinations:", len(unexpected_combinations))

if missing_combinations:
    for item in sorted(missing_combinations):
        print("MISSING:", item)

if unexpected_combinations:
    for item in sorted(unexpected_combinations):
        print("UNEXPECTED:", item)

if missing_combinations or unexpected_combinations:
    raise SystemExit(
        "\nSTOP: Experiment coverage does not match expectations."
    )

if len(selected_runs) != 100:
    raise SystemExit("STOP: Expected exactly 100 selected runs.")


# --------------------------------------------------
# 4. SAVE INDIVIDUAL RUN RESULTS
# --------------------------------------------------

fields = list(selected_runs[0].keys())

save_csv(
    OUT / "XAI_RESULTS_SUMMARY.csv",
    selected_runs,
    fields
)


# --------------------------------------------------
# 5. CALCULATE MEAN AND STANDARD DEVIATION
# --------------------------------------------------

aggregate_groups = defaultdict(list)

for r in selected_runs:
    key = (
        r["dataset_folder"],
        r["method"],
        r["backbone_class"]
    )

    aggregate_groups[key].append(r)


aggregate = []

for key, items in sorted(aggregate_groups.items()):

    items = sorted(items, key=lambda x: int(x["seed"]))

    row = {
        "dataset_folder": key[0],
        "method": key[1],
        "backbone_class": key[2],
        "n_runs": len(items),
        "seeds": ";".join(str(x["seed"]) for x in items),
    }

    for metric in METRICS:

        values = []

        for item in items:
            value = item.get(metric)

            if isinstance(value, (int, float)):
                if math.isfinite(value):
                    values.append(value)

        row[metric + "_mean"] = (
            statistics.mean(values) if values else None
        )

        row[metric + "_std"] = (
            statistics.stdev(values)
            if len(values) > 1 else None
        )

    aggregate.append(row)


save_csv(
    OUT / "XAI_MEAN_STD.csv",
    aggregate,
    list(aggregate[0].keys())
)


# --------------------------------------------------
# 6. SAVE SHAP / LIME RESULTS
# --------------------------------------------------

xai_columns = [
    "dataset_folder",
    "method",
    "seed",
    "shap_additivity_pass_rate",
    "mean_lime_local_r2",
    "mean_shap_lime_rank_spearman",
    "mean_shap_stability_spearman",
    "mean_lime_stability_spearman",
]

save_csv(
    OUT / "XAI_SHAP_LIME_SUMMARY.csv",
    selected_runs,
    xai_columns
)


# --------------------------------------------------
# 7. PRESERVE RESNET18 VALIDATED RESULTS
# --------------------------------------------------

resnet_file = Path(
    "data/resnet18_cifar10/heads_seed_42.csv"
)

if resnet_file.exists():

    with open(resnet_file) as f:
        resnet_rows = list(csv.DictReader(f))

    if resnet_rows:
        save_csv(
            OUT / "RESNET18_VALIDATED_SEED42.csv",
            resnet_rows,
            list(resnet_rows[0].keys())
        )


# --------------------------------------------------
# 8. FINAL REPORT
# --------------------------------------------------

print("\n========== AGGREGATION COMPLETE ==========")

print("Selected individual runs:", len(selected_runs))
print("Dataset count:", len(expected_datasets))
print("Methods per dataset:", len(expected_methods))
print("Seeds per method:", len(EXPECTED_SEEDS))
print("Mean/std groups:", len(aggregate_groups))

print("\nOUTPUT FILES:")

for p in sorted(OUT.glob("*.csv")):
    with open(p) as f:
        count = sum(1 for _ in f) - 1

    print(f"{p}: {count} rows")

print("\nALL VALIDATION CHECKS PASSED.")
