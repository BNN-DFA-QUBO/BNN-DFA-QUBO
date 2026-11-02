"""Shared-representation Stroke classifier-head comparison with validation tuning.

This runner trains one BNN+DFA backbone per seed (or reuses a saved representation),
freezes it, extracts representations for train, validation, and test splits, and evaluates
all classifier heads on the exact same representations:
1. Real LS
2. Binarized Ridge LS
3. Direct Binary
4. STE Binary (validation-only checkpointing)
5. QUBO Binary (validation-tuned or fixed)

Rules:
- All heads share the identical frozen representation for a given seed.
- Model selection and hyperparameter tuning use validation data ONLY.
- Official test partition is evaluated only once for final reporting.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import torch

from utils.stroke_config import (
    BALANCING_METHOD,
    SEEDS,
    SPLIT_SEED,
    STE_EPOCHS,
    QUBO_NUM_READS,
    QUBO_NUM_SWEEPS,
)
from utils.stroke_data import DATA_DIR, load_stroke_metadata, prepare_stroke_data
from utils.stroke_representation import (
    load_stroke_head_representation,
    save_representation,
)
from utils.stroke_binary_features import standardize_hidden_features
from utils.stroke_metrics import find_best_threshold, calculate_metrics
from utils.seed import set_seed
from utils.run_provenance import collect_run_provenance, stable_run_id

from experiments.stroke_experiments.stroke_bnn_dfa import train_dfa
from experiments.stroke_experiments.stroke_real_ls import train_real_ls
from experiments.stroke_experiments.stroke_dfa_binarized_ls import train_binarized_ls
from experiments.stroke_experiments.stroke_dfa_direct_binary import train_direct_binary
from experiments.stroke_experiments.stroke_dfa_ste_binary import train_ste_binary
from experiments.stroke_experiments.stroke_dfa_qubo import train_qubo
from experiments.stroke_experiments.stroke_results_utils import append_result as append_stroke_result
from experiments.stroke_experiments.tune_qubo_validation import (
    DEFAULT_NUM_READS,
    DEFAULT_NUM_SWEEPS,
    DEFAULT_SCALE_MULTIPLIERS,
    _parse_grid,
    tune_from_representations,
    write_tuning_csv,
)

DEFAULT_RESULTS_DIR = Path("data/results")
DEFAULT_DIAGNOSTIC_DIR = Path("data/results_diagnostic")


def load_or_train_stroke_representation(
    seed: int,
    split_seed: int = SPLIT_SEED,
    balancing_method: str = BALANCING_METHOD,
    output_dir: Path | str = DATA_DIR,
    overwrite: bool = False,
) -> Tuple[Dict[str, Any], Path, str]:
    output_dir = Path(output_dir)
    rep_path = output_dir / f"stroke_dfa_representation_seed_{seed}.pt"

    if rep_path.exists() and not overwrite:
        print(f"Loading existing cached representation: {rep_path}")
        data = load_stroke_head_representation(rep_path)
        return data, rep_path, "reused"

    print(f"Training fresh BNN-DFA backbone representation for seed {seed}...")
    dfa_result = train_dfa(
        seed=seed,
        split_seed=split_seed,
        balance_method=balancing_method,
        overwrite=overwrite,
        output_dir=output_dir,
    )
    artifact = save_representation(
        dfa_result,
        str(rep_path),
        feature_names=dfa_result["feature_names"],
    )
    data = load_stroke_head_representation(rep_path)
    return data, rep_path, "trained"


def run_stroke_shared_validation(
    seed: int = 42,
    split_seed: int = SPLIT_SEED,
    balancing_method: str = BALANCING_METHOD,
    output_dir: Path | str = DATA_DIR,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    fixed_qubo: bool = False,
    num_reads_grid: tuple = DEFAULT_NUM_READS,
    num_sweeps_grid: tuple = DEFAULT_NUM_SWEEPS,
    scale_multipliers: tuple = DEFAULT_SCALE_MULTIPLIERS,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    set_seed(seed)
    output_dir = Path(output_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = DEFAULT_DIAGNOSTIC_DIR
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare data & ensure split is consistent
    prepare_stroke_data(
        split_seed=split_seed,
        balance_method=balancing_method,
        overwrite=overwrite,
        output_dir=output_dir,
    )
    preprocessing_metadata = load_stroke_metadata(output_dir / "metadata.json")

    # 2. Train or load frozen backbone representation
    rep_data, rep_path, rep_source = load_or_train_stroke_representation(
        seed=seed,
        split_seed=split_seed,
        balancing_method=balancing_method,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    run_provenance = collect_run_provenance()
    rep_provenance = rep_data.get("run_provenance", {})

    backbone_training_time = float(rep_data.get("training_seconds", 0.0))
    feature_extraction_time = float(rep_data.get("feature_extraction_seconds", 0.0))
    current_backbone_time = backbone_training_time if rep_source == "trained" else 0.0
    current_extraction_time = feature_extraction_time if rep_source == "trained" else 0.0

    # Backbone parameters: 15 input features -> 15*64 + 64 = 1024; 64*1 + 1 = 65 -> 1089
    backbone_param_count = 15 * 64 + 64 + 64 * 1 + 1

    H_train_raw = rep_data["H_train"]
    H_val_raw = rep_data["H_val"]
    H_test_raw = rep_data["H_test"]
    y_train = torch.as_tensor(rep_data["y_train"]).float().reshape(-1)
    y_val = torch.as_tensor(rep_data["y_val"]).float().reshape(-1)
    y_test = torch.as_tensor(rep_data["y_test"]).float().reshape(-1)

    print("\n" + "=" * 70)
    print(f"STROKE SHARED-BACKBONE EVALUATION | Seed: {seed}")
    print(f"Representation: {rep_path.name} ({rep_source})")
    print(f"Features: train={tuple(H_train_raw.shape)}, val={tuple(H_val_raw.shape)}, test={tuple(H_test_raw.shape)}")
    print("=" * 70)

    rows: List[Dict[str, Any]] = []

    # Head 1: Real LS
    print("\n--- 1. Real Least Squares ---")
    t0 = time.perf_counter()
    real_ls_res = train_real_ls(rep_path)
    real_ls_fit_time = time.perf_counter() - t0
    real_ls_param_count = 65  # 64 float weights + 1 intercept
    real_ls_res.update({
        "method": "Real LS",
        "head_fit_time_sec": real_ls_fit_time,
        "head_parameter_count": real_ls_param_count,
        "configuration": "numpy.linalg.lstsq closed-form; validation threshold",
    })
    rows.append(real_ls_res)

    # Head 2: Binarized Ridge LS (refit scale+bias)
    print("\n--- 2. Binarized Ridge LS (refit scale+bias) ---")
    t0 = time.perf_counter()
    bin_ridge_res = train_binarized_ls(rep_path, ls_solver="ridge")
    bin_ridge_fit_time = time.perf_counter() - t0
    bin_ridge_param_count = 66  # 64 binary weights + 1 scale + 1 intercept
    bin_ridge_res.update({
        "method": "Binarized Ridge LS (refit scale+bias)",
        "head_fit_time_sec": bin_ridge_fit_time,
        "head_parameter_count": bin_ridge_param_count,
        "configuration": "standardized ridge SVD -> sign binarize -> training scale/intercept refit",
    })
    rows.append(bin_ridge_res)

    # Head 3: Direct Binary
    print("\n--- 3. Direct Binary ---")
    t0 = time.perf_counter()
    direct_res = train_direct_binary(rep_path, seed)
    direct_fit_time = time.perf_counter() - t0
    direct_param_count = 64  # 64 binary weights (bias fixed at 0)
    direct_res.update({
        "method": "Direct Binary",
        "head_fit_time_sec": direct_fit_time,
        "head_parameter_count": direct_param_count,
        "configuration": "sign of positive-minus-negative training mean",
    })
    rows.append(direct_res)

    # Head 4: STE Binary
    print("\n--- 4. STE Binary ---")
    t0 = time.perf_counter()
    ste_res = train_ste_binary(rep_path, seed)
    ste_fit_time = time.perf_counter() - t0
    ste_param_count = 65  # 64 deployed binary weights + 1 bias
    ste_res.update({
        "method": "STE Binary",
        "head_fit_time_sec": ste_fit_time,
        "head_parameter_count": ste_param_count,
        "configuration": f"Adam STE; validation-selected checkpoint; init={ste_res.get('initialization', 'unknown')}",
    })
    rows.append(ste_res)

    # Head 5: QUBO Binary (Tuned or Fixed)
    print("\n--- 5. QUBO Binary ---")
    if fixed_qubo:
        print("Running fixed QUBO head...")
        t0 = time.perf_counter()
        qubo_res = train_qubo(rep_path, seed)
        qubo_fit_time = float(qubo_res.get("qubo_search_seconds", time.perf_counter() - t0))
        qubo_param_count = 66
        qubo_res.update({
            "method": "QUBO Binary",
            "head_fit_time_sec": qubo_fit_time,
            "head_parameter_count": qubo_param_count,
            "configuration": f"fixed QUBO; num_reads={QUBO_NUM_READS}; num_sweeps={QUBO_NUM_SWEEPS}",
            "qubo_grid_protocol": "fixed_qubo",
        })
        rows.append(qubo_res)
    else:
        print("Running validation-tuned QUBO grid search...")
        H_train_s, H_val_s, H_test_s, transform = standardize_hidden_features(
            H_train_raw, H_val_raw, H_test_raw
        )
        t0 = time.perf_counter()
        winner, tuning_rows = tune_from_representations(
            H_train=H_train_s,
            y_train=y_train,
            H_val=H_val_s,
            y_val=y_val,
            seed=seed,
            num_reads_grid=num_reads_grid,
            num_sweeps_grid=num_sweeps_grid,
            scale_multipliers=scale_multipliers,
        )
        total_qubo_tuning_time = time.perf_counter() - t0
        if winner is None:
            raise RuntimeError("QUBO tuning failed to select any configuration")

        tuning_csv_path = write_tuning_csv(
            rows=tuning_rows,
            seed=seed,
            output_dir=diagnostic_dir,
            artifact_id=rep_data.get("representation_sha256", "")[:12],
            n_train=len(H_train_s),
            n_val=len(H_val_s),
            n_test=len(H_test_s),
        )
        print(f"QUBO tuning diagnostics logged to: {tuning_csv_path}")

        # Final test evaluation on the selected configuration strictly once
        test_scores = winner["_alpha"] * (H_test_s @ winner["_weights"]) + winner["_bias"]
        test_metrics = calculate_metrics(y_test.numpy(), test_scores.numpy(), winner["_threshold"])

        qubo_param_count = 66
        qubo_res = {
            "method": "QUBO Binary",
            "seed": seed,
            "head_seed": seed,
            "threshold": winner["_threshold"],
            "validation_balanced_accuracy": winner["validation_balanced_accuracy"],
            "validation_selection_score": winner["validation_selection_score"],
            "validation_f1": winner["validation_f1"],
            "validation_pr_auc": winner["validation_pr_auc"],
            "validation_roc_auc": winner["validation_roc_auc"],
            "alpha": float(winner["_alpha"]),
            "base_alpha": float(winner["base_alpha"]),
            "alpha_multiplier": float(winner["scale_multiplier"]),
            "bias": float(winner["_bias"]),
            "qubo_num_reads": winner["num_reads"],
            "qubo_num_sweeps": winner["num_sweeps"],
            "qubo_solver_seed_selected": winner["solver_seed"],
            "qubo_candidate_count": winner["candidate_count"],
            "qubo_energy": winner["energy"],
            "qubo_candidate_source": winner["candidate_source"],
            "qubo_time_sec": total_qubo_tuning_time,
            "qubo_selected_candidate_time_sec": winner["runtime_sec"],
            "qubo_grid_protocol": "fixed_27_configuration_v1",
            "head_fit_time_sec": total_qubo_tuning_time,
            "head_parameter_count": qubo_param_count,
            "configuration": (
                f"validation-tuned: num_reads={winner['num_reads']}, "
                f"num_sweeps={winner['num_sweeps']}, scale_multiplier={winner['scale_multiplier']}"
            ),
            "hidden_feature_transform": transform["transform"],
            "standardized_constant_units": transform["constant_units"],
            **test_metrics,
        }
        rows.append(qubo_res)

    # 3. Post-process and populate common provenance/timing schema across all rows
    formatted_results = []
    for r in rows:
        method_name = r["method"]
        head_fit_sec = float(r.get("head_fit_time_sec", 0.0))
        head_param_cnt = int(r.get("head_parameter_count", 65))
        total_param_cnt = backbone_param_count + head_param_cnt

        cur_total_time = current_backbone_time + current_extraction_time + head_fit_sec
        e2e_time = backbone_training_time + feature_extraction_time + head_fit_sec

        entry = dict(r)
        entry["dataset"] = "stroke"
        entry["experiment_type"] = "shared_backbone_head_comparison"
        entry["seed"] = seed
        entry["model_seed"] = seed
        entry["training_seed"] = seed
        entry["representation_seed"] = rep_data.get("training_seed", seed)
        entry["split_seed"] = preprocessing_metadata["split_seed"]
        entry["preprocessing_version"] = preprocessing_metadata["preprocessing_version"]
        entry["preprocessing_config_hash"] = preprocessing_metadata["preprocessing_config_hash"]
        entry["dataset_sha256"] = preprocessing_metadata["dataset_sha256"]
        entry["balancing_method"] = preprocessing_metadata["balancing_method"]
        entry["representation_path"] = str(rep_path)
        entry["representation_sha256"] = rep_data.get("representation_sha256", "")
        entry["representation_cache_status"] = rep_source
        entry["backbone_training_time_sec"] = current_backbone_time
        entry["feature_extraction_time_sec"] = current_extraction_time
        entry["head_fit_time_sec"] = head_fit_sec
        entry["current_run_total_time_sec"] = cur_total_time
        entry["end_to_end_training_time_sec"] = e2e_time
        entry["pipeline_timing_status"] = "complete"
        entry["backbone_parameter_count"] = backbone_param_count
        entry["head_parameter_count"] = head_param_cnt
        entry["total_parameter_count"] = total_param_cnt
        entry["status"] = entry.get("status", "ok")
        entry["failure_reason"] = entry.get("failure_reason", "")

        entry["run_id"] = stable_run_id("stroke_shared_backbone", {
            "method": method_name,
            "seed": seed,
            "representation_sha256": rep_data.get("representation_sha256", ""),
            "qubo_num_reads": entry.get("qubo_num_reads"),
            "qubo_num_sweeps": entry.get("qubo_num_sweeps"),
            "alpha_multiplier": entry.get("alpha_multiplier"),
        }, run_provenance)

        entry["git_commit_sha"] = run_provenance.get("git_commit_sha", "unavailable")
        entry["git_worktree_dirty"] = run_provenance.get("git_worktree_dirty", "unavailable")
        entry["git_worktree_fingerprint"] = run_provenance.get("git_worktree_fingerprint", "unavailable")
        entry["python_version"] = run_provenance.get("python_version", "unavailable")
        entry["os_platform"] = run_provenance.get("os_platform", "unavailable")

        formatted_results.append(entry)

    # Save summary table for this run
    csv_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = diagnostic_dir / f"stroke_shared_validation_heads_seed_{seed}_{csv_timestamp}.csv"
    _save_shared_results_csv(formatted_results, summary_csv)
    print(f"\nDetailed shared-backbone evaluation saved to: {summary_csv}")

    # Also append/save to canonical results path
    canonical_csv = results_dir / "stroke_shared_results.csv"
    _append_canonical_results(formatted_results, canonical_csv)
    print(f"Appended to canonical results log: {canonical_csv}")

    # Also log to main shared stroke_results.csv
    for r in formatted_results:
        append_stroke_result(
            experiment="stroke_shared_validation",
            method=r["method"],
            seed=seed,
            balanced_accuracy=r.get("balanced_accuracy", 0.0),
            f1_score=r.get("f1"),
            precision=r.get("precision"),
            recall=r.get("recall"),
            roc_auc=r.get("roc_auc"),
            pr_auc=r.get("pr_auc"),
            validation_balanced_accuracy=r.get("validation_balanced_accuracy"),
            threshold=r.get("threshold"),
            backbone_training_time_sec=r.get("backbone_training_time_sec"),
            feature_extraction_time_sec=r.get("feature_extraction_time_sec"),
            head_fit_time_sec=r.get("head_fit_time_sec"),
            current_run_total_time_sec=r.get("current_run_total_time_sec"),
            end_to_end_training_time_sec=r.get("end_to_end_training_time_sec"),
            backbone_parameter_count=r.get("backbone_parameter_count"),
            head_parameter_count=r.get("head_parameter_count"),
            total_parameter_count=r.get("total_parameter_count"),
            parameter_count=r.get("backbone_parameter_count"),
            qubo_tuning_enabled=not fixed_qubo if "QUBO" in r["method"] else False,
            qubo_num_reads_selected=r.get("qubo_num_reads"),
            qubo_num_sweeps_selected=r.get("qubo_num_sweeps"),
            qubo_scale_multiplier_selected=r.get("alpha_multiplier"),
            qubo_time_sec=r.get("qubo_time_sec"),
            qubo_selected_candidate_time_sec=r.get("qubo_selected_candidate_time_sec"),
            representation_artifact_id=r.get("representation_sha256", ""),
            representation_path=str(r.get("representation_path", "")),
            representation_cache_status=r.get("representation_cache_status", ""),
            representation_sha256=r.get("representation_sha256", ""),
            notes="shared backbone validation protocol",
        )

    print("\n" + "=" * 70)
    print(f"{'Method':<35} | {'Val BA %':<10} | {'Test BA %':<10} | {'F1 %':<8} | {'ROC %':<8} | {'Time (s)':<8}")
    print("-" * 70)
    for r in formatted_results:
        val_ba = float(r.get("validation_balanced_accuracy", 0.0)) * 100
        test_ba = float(r.get("balanced_accuracy", 0.0)) * 100
        f1 = float(r.get("f1", 0.0)) * 100
        roc = float(r.get("roc_auc", 0.0)) * 100
        t_fit = float(r.get("head_fit_time_sec", 0.0))
        print(f"{r['method']:<35} | {val_ba:<10.2f} | {test_ba:<10.2f} | {f1:<8.2f} | {roc:<8.2f} | {t_fit:<8.2f}")
    print("=" * 70)

    return formatted_results


def _save_shared_results_csv(results: List[Dict[str, Any]], path: Path):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def _append_canonical_results(results: List[Dict[str, Any]], path: Path):
    if not results:
        return
    path = Path(path)
    file_exists = path.exists()

    fieldnames = [
        "dataset",
        "experiment_type",
        "method",
        "status",
        "failure_reason",
        "seed",
        "model_seed",
        "training_seed",
        "representation_seed",
        "split_seed",
        "balanced_accuracy",
        "validation_balanced_accuracy",
        "validation_selection_score",
        "f1",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "threshold",
        "head_fit_time_sec",
        "backbone_training_time_sec",
        "feature_extraction_time_sec",
        "current_run_total_time_sec",
        "end_to_end_training_time_sec",
        "pipeline_timing_status",
        "backbone_parameter_count",
        "head_parameter_count",
        "total_parameter_count",
        "qubo_num_reads",
        "qubo_num_sweeps",
        "alpha_multiplier",
        "alpha",
        "base_alpha",
        "bias",
        "qubo_time_sec",
        "qubo_selected_candidate_time_sec",
        "qubo_grid_protocol",
        "configuration",
        "representation_cache_status",
        "representation_path",
        "representation_sha256",
        "run_id",
        "git_commit_sha",
    ]

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for BNN-DFA backbone & heads")
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--balancing", choices=("random_oversample", "none"), default=BALANCING_METHOD)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--fixed-qubo", action="store_true", help="Use fixed QUBO instead of validation grid tuning")
    parser.add_argument(
        "--num-reads-grid",
        type=lambda s: _parse_grid(s, int, "num-reads-grid"),
        default=DEFAULT_NUM_READS,
    )
    parser.add_argument(
        "--num-sweeps-grid",
        type=lambda s: _parse_grid(s, int, "num-sweeps-grid"),
        default=DEFAULT_NUM_SWEEPS,
    )
    parser.add_argument(
        "--scale-multipliers",
        type=lambda s: _parse_grid(s, float, "scale-multipliers"),
        default=DEFAULT_SCALE_MULTIPLIERS,
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    run_stroke_shared_validation(
        seed=args.seed,
        split_seed=args.split_seed,
        balancing_method=args.balancing,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        fixed_qubo=args.fixed_qubo,
        num_reads_grid=args.num_reads_grid,
        num_sweeps_grid=args.num_sweeps_grid,
        scale_multipliers=args.scale_multipliers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

