"""Paired Stroke comparison of legacy LS and Ridge-LS classifier heads.

For each representation seed, all four heads use exactly the same saved
H_train/H_val/H_test tensors. Thresholds are selected using validation scores;
the test split is used only for final metrics. Results are written to fresh,
timestamped CSVs and do not touch the canonical Stroke result files.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from models.ls_heads import fit_ridge_regression
from utils.stroke_config import BALANCING_METHOD, SEEDS, SPLIT_SEED
from utils.stroke_data import DATA_DIR
from sklearn.metrics import balanced_accuracy_score
from utils.stroke_metrics import calculate_metrics
from utils.stroke_representation import load_representation, save_representation
from experiments.stroke_experiments.stroke_bnn_dfa import train_dfa


METRICS = (
    "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "tn", "fp", "fn", "tp", "predicted_positive", "actual_positive",
)


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def fit_legacy_ls(H_train, y_train):
    """Historical Stroke Real-LS fit: NumPy least squares on {0,1} labels."""
    H = _as_numpy(H_train).astype(np.float64, copy=False)
    y = _as_numpy(y_train).reshape(-1).astype(np.float64, copy=False)
    design = np.column_stack((H, np.ones(len(H), dtype=np.float64)))
    weights, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return weights


def score_continuous(H, weights):
    H = _as_numpy(H).astype(np.float64, copy=False)
    return np.column_stack((H, np.ones(len(H), dtype=np.float64))) @ weights


def fit_legacy_binarized_ls(H_train, y_train):
    """Historical Stroke Binarized-LS: sign weights, retain LS intercept."""
    weights = fit_legacy_ls(H_train, y_train)
    binary = np.sign(weights[:-1])
    binary[binary == 0] = 1.0
    return binary, float(weights[-1])


def fit_ridge_ls(H_train, y_train, lambda_relative):
    """Stable float64 Ridge LS on {0,1} stroke targets with an intercept."""
    target = torch.as_tensor(_as_numpy(y_train).reshape(-1), dtype=torch.float64)
    weights = fit_ridge_regression(
        torch.as_tensor(_as_numpy(H_train), dtype=torch.float64),
        target,
        lambda_relative=lambda_relative,
    )[:, 0].numpy()
    return weights


def fit_ridge_binarized_ls(H_train, y_train, weights):
    """Sign Ridge-LS feature weights, then refit one scale and intercept.

    The scale/intercept solve uses training representations and {0,1} labels
    only. Zero signs are mapped to +1, matching the existing Stroke baseline.
    """
    H = _as_numpy(H_train).astype(np.float64, copy=False)
    y = _as_numpy(y_train).reshape(-1).astype(np.float64, copy=False)
    binary = np.sign(weights[:-1]).astype(np.float64)
    binary[binary == 0] = 1.0

    z = H @ binary
    z_mean = float(z.mean())
    y_mean = float(y.mean())
    z_centered = z - z_mean
    y_centered = y - y_mean
    denominator = float(z_centered @ z_centered)
    scale_raw = (
        float(z_centered @ y_centered) / denominator
        if denominator > 1e-12 else 0.0
    )

    # Keep a non-negative scale without changing fitted training scores.
    if scale_raw < 0:
        binary *= -1.0
        scale = -scale_raw
    else:
        scale = scale_raw
    bias = y_mean - scale_raw * z_mean
    return binary, float(scale), float(bias)


def find_best_threshold_exact(y_true, scores):
    """Select a validation threshold by checking every score boundary.

    Unlike a fixed linear grid, unique-score boundaries cannot skip useful
    thresholds when score ranges contain extreme values or large gaps.
    Test labels are never passed to this function.
    """
    y = np.asarray(y_true).reshape(-1).astype(np.int64)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y) == 0 or len(y) != len(values):
        raise ValueError("y_true and scores must be non-empty and have equal length")
    if not np.isfinite(values).all():
        raise ValueError("Threshold selection requires finite scores")

    candidates = np.unique(values)
    candidates = np.concatenate(
        (candidates, [np.nextafter(candidates[-1], np.inf)])
    )
    best_threshold = float(candidates[0])
    best_score = -np.inf
    for threshold in candidates:
        score = balanced_accuracy_score(y, values >= threshold)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def evaluate_scores(y_val, validation_scores, y_test, test_scores):
    threshold, validation_ba = find_best_threshold_exact(y_val, validation_scores)
    metrics = calculate_metrics(y_test, test_scores, threshold)
    return {
        "threshold": float(threshold),
        "validation_balanced_accuracy": float(validation_ba),
        **metrics,
    }


def load_or_train_representation(seed, args, artifacts_dir):
    if args.representation_dir is not None:
        representation_path = (
            Path(args.representation_dir) / f"stroke_dfa_representation_seed_{seed}.pt"
        )
        if not representation_path.is_file():
            raise FileNotFoundError(
                f"Missing representation for seed {seed}: {representation_path}. "
                "Omit --representation-dir to train fresh representations."
            )
        data = load_representation(representation_path)
        metadata = data.get("data_metadata", {})
        if int(data.get("split_seed", -1)) != args.split_seed:
            raise ValueError(
                f"Seed {seed} representation uses split seed {data.get('split_seed')}, "
                f"not requested split seed {args.split_seed}."
            )
        if metadata.get("balancing_method") != args.balancing:
            raise ValueError(
                f"Seed {seed} representation uses balancing={metadata.get('balancing_method')}, "
                f"not requested balancing={args.balancing}."
            )
        return representation_path, data, "reused"

    representation_path = artifacts_dir / f"stroke_dfa_representation_seed_{seed}.pt"
    result = train_dfa(
        seed=seed,
        split_seed=args.split_seed,
        balance_method=args.balancing,
        overwrite=False,
        output_dir=artifacts_dir,
    )
    save_representation(result, str(representation_path), feature_names=result["feature_names"])
    data = load_representation(representation_path)
    return representation_path, data, "trained"


def compare_representation(seed, data, args, representation_path, representation_source):
    H_train = data["H_train"]
    H_val = data["H_val"]
    H_test = data["H_test"]
    y_train = _as_numpy(data["y_train"]).reshape(-1).astype(np.int64)
    y_val = _as_numpy(data["y_val"]).reshape(-1).astype(np.int64)
    y_test = _as_numpy(data["y_test"]).reshape(-1).astype(np.int64)

    rows = []
    candidates = []

    started = time.perf_counter()
    legacy_weights = fit_legacy_ls(H_train, y_train)
    legacy_fit_seconds = time.perf_counter() - started
    candidates.append((
        "Real LS (legacy NumPy)",
        score_continuous(H_val, legacy_weights),
        score_continuous(H_test, legacy_weights),
        legacy_fit_seconds,
        "unregularized NumPy float64 least squares; {0,1} targets; fitted intercept",
    ))

    started = time.perf_counter()
    ridge_weights = fit_ridge_ls(H_train, y_train, args.lambda_relative)
    ridge_fit_seconds = time.perf_counter() - started
    candidates.append((
        "Ridge LS (regularized SVD)",
        score_continuous(H_val, ridge_weights),
        score_continuous(H_test, ridge_weights),
        ridge_fit_seconds,
        f"float64 augmented SVD with ridge regularization; lambda_relative={args.lambda_relative}; {{0,1}} targets; unpenalized intercept",
    ))

    started = time.perf_counter()
    old_binary, old_bias = fit_legacy_binarized_ls(H_train, y_train)
    legacy_binary_fit_seconds = time.perf_counter() - started
    old_val_scores = _as_numpy(H_val).astype(np.float64) @ old_binary + old_bias
    old_test_scores = _as_numpy(H_test).astype(np.float64) @ old_binary + old_bias
    candidates.append((
        "Binarized LS (legacy)",
        old_val_scores,
        old_test_scores,
        legacy_binary_fit_seconds,
        "unregularized NumPy float64 LS; sign feature weights; retains continuous LS intercept",
    ))

    started = time.perf_counter()
    ridge_binary, ridge_scale, ridge_bias = fit_ridge_binarized_ls(
        H_train, y_train, ridge_weights
    )
    ridge_binary_fit_seconds = time.perf_counter() - started
    ridge_val_scores = (
        _as_numpy(H_val).astype(np.float64) @ ridge_binary
    ) * ridge_scale + ridge_bias
    ridge_test_scores = (
        _as_numpy(H_test).astype(np.float64) @ ridge_binary
    ) * ridge_scale + ridge_bias
    candidates.append((
        "Binarized Ridge LS (refit scale+bias)",
        ridge_val_scores,
        ridge_test_scores,
        ridge_fit_seconds + ridge_binary_fit_seconds,
        f"float64 Ridge LS; binary signs; refit scale+bias on training rows; lambda_relative={args.lambda_relative}",
    ))

    metadata = data.get("data_metadata", {})
    class_counts = {
        "train": metadata.get("resampled_train_class_counts", {}),
        "validation": metadata.get("validation_class_counts", {}),
        "test": metadata.get("test_class_counts", {}),
    }
    provenance = {
        "seed": seed,
        "training_seed": int(data.get("training_seed", data.get("seed", seed))),
        "split_seed": int(data.get("split_seed", args.split_seed)),
        "balancing_method": metadata.get("balancing_method", args.balancing),
        "dataset_sha256": data.get("dataset_sha256", ""),
        "preprocessing_version": data.get("preprocessing_version", ""),
        "preprocessing_config_hash": metadata.get("preprocessing_config_hash", ""),
        "input_size": data.get("input_size", ""),
        "hidden_size": data.get("hidden_size", int(H_train.shape[1])),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_val)),
        "n_test": int(len(y_test)),
        "split_class_counts": json.dumps(class_counts, sort_keys=True),
        "representation_path": str(representation_path),
        "representation_source": representation_source,
        "lambda_relative": args.lambda_relative,
    }

    for method, val_scores, test_scores, fit_seconds, description in candidates:
        metric_result = evaluate_scores(y_val, val_scores, y_test, test_scores)
        row = {
            **provenance,
            "method": method,
            "method_description": description,
            "validation_balanced_accuracy": metric_result["validation_balanced_accuracy"],
            "threshold": metric_result["threshold"],
            "fit_seconds": fit_seconds,
            **{key: metric_result[key] for key in METRICS},
        }
        rows.append(row)
        print(
            f"seed={seed:>4} | {method:<39} | "
            f"val BA={100 * row['validation_balanced_accuracy']:.3f}% | "
            f"test BA={100 * row['balanced_accuracy']:.3f}% | "
            f"F1={100 * row['f1']:.3f}% | "
            f"PR-AUC={100 * row['pr_auc']:.3f}%"
        )
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarise(rows):
    summary = []
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        row = {
            "method": method,
            "n_seeds": len(group),
        }
        for metric in ("balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fit_seconds"):
            values = [float(item[metric]) for item in group if item.get(metric) not in ("", None)]
            row[f"{metric}_mean"] = statistics.mean(values) if values else ""
            row[f"{metric}_sample_std"] = statistics.stdev(values) if len(values) > 1 else ""
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--balancing", choices=("random_oversample", "none"), default=BALANCING_METHOD)
    parser.add_argument("--lambda-relative", type=float, default=1e-4)
    parser.add_argument(
        "--representation-dir", type=Path, default=None,
        help="Reuse validated representations in this directory instead of training fresh representations.",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=None,
        help="Directory for fresh preprocessing/representation files; if omitted a timestamped directory is created.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DATA_DIR.parent / "results_diagnostic"
    raw_path = output_dir / f"stroke_ls_head_validation_{timestamp}.csv"
    summary_path = output_dir / f"stroke_ls_head_validation_summary_{timestamp}.csv"
    artifacts_dir = (
        Path(args.artifacts_dir)
        if args.artifacts_dir is not None
        else DATA_DIR.parent / f"stroke_ls_validation_artifacts_{timestamp}"
    )

    if args.lambda_relative < 0:
        parser.error("--lambda-relative must be >= 0")
    if raw_path.exists() or summary_path.exists():
        raise FileExistsError("Diagnostic result path already exists; rerun to obtain a fresh timestamp.")
    if args.representation_dir is None:
        artifacts_dir.mkdir(parents=True, exist_ok=False)

    all_rows = []
    for seed in args.seeds:
        print("\n" + "=" * 92)
        print(f"STROKE LS VALIDATION — SEED {seed}")
        print("=" * 92)
        representation_path, data, source = load_or_train_representation(
            seed, args, artifacts_dir
        )
        all_rows.extend(
            compare_representation(seed, data, args, representation_path, source)
        )

    if not all_rows:
        raise RuntimeError("No results were produced.")

    common_fields = list(all_rows[0].keys())
    write_csv(raw_path, all_rows, common_fields)
    summary_rows = summarise(all_rows)
    summary_fields = list(summary_rows[0].keys())
    write_csv(summary_path, summary_rows, summary_fields)

    print("\n" + "=" * 92)
    print("STROKE LS VALIDATION COMPLETE")
    print("=" * 92)
    print(f"Raw results: {raw_path}")
    print(f"Summary:     {summary_path}")
    if args.representation_dir is None:
        print(f"Artifacts:   {artifacts_dir}")
    print("\nMean test balanced accuracy by method:")
    for row in summary_rows:
        value = row.get("balanced_accuracy_mean")
        deviation = row.get("balanced_accuracy_sample_std")
        if value != "":
            dev_text = f" +/- {100 * deviation:.3f} pp" if deviation != "" else ""
            print(f"  {row['method']}: {100 * value:.3f}%{dev_text}")


if __name__ == "__main__":
    main()
