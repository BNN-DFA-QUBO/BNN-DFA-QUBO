"""Collect run-level XAI metrics without treating explained rows as independent seeds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = [
    "dataset_accuracy_on_test", "dataset_balanced_accuracy_on_test", "dataset_macro_f1_on_test",
    "shap_additivity_abs_error", "lime_local_r2", "shap_lime_rank_spearman",
    "shap_stability_spearman", "shap_stability_topk_jaccard",
    "lime_stability_spearman", "lime_stability_topk_jaccard",
    "shap_topk_vs_random_ratio", "lime_topk_vs_random_ratio",
]


def main():
    p = argparse.ArgumentParser(description="Aggregate XAI runs by dataset, method and seed.")
    p.add_argument("--datasets", nargs="+", choices=["bank", "mnist", "cifar10", "stroke"], required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--input-dir", type=Path, default=Path("data/xai_results"))
    p.add_argument("--output-dir", type=Path, default=Path("data/xai_results/aggregated"))
    args = p.parse_args()
    records = []
    missing = []
    for dataset in args.datasets:
        for method in ("real_ls", "binarized_ls", "direct_binary", "ste_binary", "qubo"):
            for seed in args.seeds:
                matches = sorted((args.input_dir / dataset / method).glob(f"seed_{seed}_*/run_manifest.json"))
                if not matches:
                    missing.append(f"{dataset}/{method}/seed_{seed}")
                    continue
                # Select latest timestamped run for this exact dataset/method/seed.
                manifest_path = matches[-1]
                manifest = json.loads(manifest_path.read_text())
                sample_path = manifest_path.parent / "sample_metrics.csv"
                sample = pd.read_csv(sample_path)
                if sample.empty:
                    raise SystemExit(f"Empty sample metrics: {sample_path}")
                if set(sample["seed"].unique()) != {seed}:
                    raise SystemExit(f"Seed mismatch in {sample_path}")
                if manifest.get("backend", {}).get("shap") != "shap.KernelExplainer":
                    raise SystemExit(f"Unexpected SHAP backend in {manifest_path}")
                lime_backends = set(sample["lime_backend"].dropna().astype(str))
                if not lime_backends:
                    raise SystemExit(f"LIME backend missing in {sample_path}")
                rec = {"dataset": dataset, "method": method, "seed": seed,
                       "run_directory": str(manifest_path.parent),
                       "lime_backend": ";".join(sorted(lime_backends)),
                       "representation_sha256": manifest.get("representation_sha256"),
                       "n_test": manifest.get("test_evaluation", {}).get("n_test"),
                       "n_explained": len(sample)}
                for metric in METRICS:
                    if metric in sample.columns:
                        vals = pd.to_numeric(sample[metric], errors="coerce")
                        rec[metric] = float(vals.mean())
                    elif metric in manifest.get("test_evaluation", {}):
                        rec[metric] = float(manifest["test_evaluation"][metric])
                    elif metric == "dataset_accuracy_on_test":
                        rec[metric] = float(manifest["test_evaluation"]["accuracy"])
                    elif metric == "dataset_balanced_accuracy_on_test":
                        rec[metric] = float(manifest["test_evaluation"]["balanced_accuracy"])
                    elif metric == "dataset_macro_f1_on_test":
                        rec[metric] = float(manifest["test_evaluation"]["macro_f1"])
                rec["shap_additivity_pass_rate"] = float((sample["shap_local_accuracy_status"] == "pass").mean())
                records.append(rec)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_df = pd.DataFrame(records)
    run_file = args.output_dir / "xai_metrics_by_seed.csv"
    run_df.to_csv(run_file, index=False)
    summary_rows = []
    if not run_df.empty:
        for (dataset, method), group in run_df.groupby(["dataset", "method"], sort=True):
            row = {"dataset": dataset, "method": method, "n_seeds": int(group["seed"].nunique())}
            for metric in METRICS + ["shap_additivity_pass_rate"]:
                if metric in group:
                    values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
                    row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
                    row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                else:
                    row[f"{metric}_mean"] = np.nan
                    row[f"{metric}_sd"] = np.nan
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_file = args.output_dir / "xai_metrics_mean_sd.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved per-seed metrics: {run_file}")
    print(f"Saved mean ± sample SD: {summary_file}")
    if missing:
        print("\nMISSING RUNS (not fabricated):")
        for item in missing:
            print(" - " + item)
    print(f"\nCollected {len(run_df)} run(s); expected {len(args.datasets)*5*len(args.seeds)}.")


if __name__ == "__main__":
    main()
