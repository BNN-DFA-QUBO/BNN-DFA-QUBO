"""Bounded validation-only grid tuning for the MNIST QUBO head.

Example: python -m experiments.MNIST_experiments.tune_qubo_validation --seed 123
Solver seed for grid entry i (zero-based): (backbone_seed + 1009 * (i + 1)) mod 2**32.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from experiments.MNIST_experiments.bnn_dfa_qubo import calculate_final_scale_and_bias, create_binary_targets
from experiments.MNIST_experiments.mnist_representation import load_or_train_mnist_representation
from models.qubo_head import calculate_ridge_ls_scales, optimize_binary_classifier
from utils.MNIST.data import get_mnist_loaders
from utils.MNIST.seed import set_seed
from utils.MNIST.representation import DEFAULT_REPRESENTATION_DIR
from utils.run_provenance import collect_run_provenance, stable_run_id

DEFAULT_NUM_READS = (25, 50, 100)
DEFAULT_NUM_SWEEPS = (250, 500, 1000)
DEFAULT_SCALE_MULTIPLIERS = (0.5, 1.0, 2.0)


def _parse_grid(raw: str, cast, name: str) -> tuple:
    try:
        values = tuple(cast(part.strip()) for part in raw.split(","))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated valid values") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} must contain positive values")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} must not contain duplicates")
    return values


def validate_grid(num_reads_grid, num_sweeps_grid, scale_multipliers):
    reads = tuple(num_reads_grid)
    sweeps = tuple(num_sweeps_grid)
    multipliers = tuple(scale_multipliers)
    if not reads or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in reads):
        raise ValueError("num_reads grid must contain positive integers")
    if not sweeps or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in sweeps):
        raise ValueError("num_sweeps grid must contain positive integers")
    if not multipliers or any(not np.isfinite(v) or v <= 0 for v in multipliers):
        raise ValueError("scale multipliers must contain positive finite numbers")
    if len(set(reads)) != len(reads) or len(set(multipliers)) != len(multipliers):
        raise ValueError("grid values must not contain duplicates")
    if len(set(sweeps)) != len(sweeps):
        raise ValueError("num_sweeps grid must not contain duplicates")
    if len(reads) * len(sweeps) * len(multipliers) != 27:
        raise ValueError("expanded QUBO grid must contain exactly 27 configurations")
    if reads != DEFAULT_NUM_READS or sweeps != DEFAULT_NUM_SWEEPS or multipliers != DEFAULT_SCALE_MULTIPLIERS:
        raise ValueError("expanded QUBO grid must use the fixed reads, sweeps, and scale values")
    return tuple((n, s, float(m)) for n in reads for s in sweeps for m in multipliers)


def select_configuration(rows):
    """Tie-break by validation accuracy, reads, multiplier distance, sweeps, multiplier."""
    if not rows:
        raise ValueError("cannot select from an empty result set")
    return min(rows, key=lambda row: (
        -float(row["validation_accuracy"]), int(row["num_reads"]),
        abs(float(row["scale_multiplier"]) - 1.0), int(row["num_sweeps"]),
        float(row["scale_multiplier"]),
    ))


def apply_scale_multiplier(base_scales, multiplier):
    return base_scales * float(multiplier)


def _accuracy(scores, labels):
    return float((scores.argmax(1) == labels).float().mean().item() * 100.0)


def tune_from_representations(H_train, y_train, H_val, y_val, seed,
                              num_reads_grid=DEFAULT_NUM_READS,
                              num_sweeps_grid=DEFAULT_NUM_SWEEPS,
                              scale_multipliers=DEFAULT_SCALE_MULTIPLIERS):
    """Fit the finite candidate grid from training features and select on validation only."""
    tuning_started = time.perf_counter()
    configurations = validate_grid(num_reads_grid, num_sweeps_grid, scale_multipliers)
    targets_train = create_binary_targets(y_train, num_classes=10)
    base_scales = calculate_ridge_ls_scales(H_train, y_train, num_classes=10, lambda_relative=1e-4)
    base_scales = torch.where(base_scales > 1e-8, base_scales, torch.full_like(base_scales, 1e-8))
    rows = []
    for index, (num_reads, num_sweeps, multiplier) in enumerate(configurations):
        solver_seed = (int(seed) + 1009 * (index + 1)) % (2 ** 32)
        started = time.perf_counter()
        row = {"num_reads": num_reads, "scale_multiplier": multiplier,
               "num_sweeps": num_sweeps, "solver_seed": solver_seed,
               "validation_accuracy": "", "selected": False,
               "status": "failed", "error": "", "runtime_sec": 0.0}
        try:
            weights = optimize_binary_classifier(
                H_train, targets_train, apply_scale_multiplier(base_scales, multiplier),
                num_classes=10, num_reads=num_reads, num_sweeps=num_sweeps, seed=solver_seed,
            )
            weights, final_scales, final_bias = calculate_final_scale_and_bias(
                H_train, y_train, weights, num_classes=10,
            )
            val_scores = H_val.float() @ (weights * final_scales.unsqueeze(0)) + final_bias
            row.update({"validation_accuracy": _accuracy(val_scores, y_val),
                        "status": "ok", "_weights": weights,
                        "_scales": final_scales, "_bias": final_bias})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["runtime_sec"] = time.perf_counter() - started
        rows.append(row)
        validation_text = (f"{row['validation_accuracy']:.2f}%"
                           if row["status"] == "ok" else f"FAILED ({row['error']})")
        print(f"num_reads={num_reads}, num_sweeps={num_sweeps}, scale_multiplier={multiplier:g}: "
              f"validation_accuracy={validation_text} runtime={row['runtime_sec']:.2f}s")
    successful = [row for row in rows if row["status"] == "ok"]
    winner = select_configuration(successful) if successful else None
    tuning_total_seconds = time.perf_counter() - tuning_started
    if winner is not None:
        winner["selected"] = True
        winner["selected_candidate_runtime_sec"] = winner["runtime_sec"]
        winner["tuning_total_seconds"] = tuning_total_seconds
    for row in rows:
        row["tuning_total_seconds"] = tuning_total_seconds
        row["selected_candidate_runtime_sec"] = (
            winner["runtime_sec"] if winner is not None and row is winner else ""
        )
    return winner, rows


def write_tuning_csv(rows, seed, epochs, validation_fraction,
                     output_dir=Path("data/results_diagnostic"),
                     representation_artifact_id="", n_train=None,
                     n_validation=None, n_test=None, run_id=None, provenance=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"qubo_validation_tuning_27config_seed_{seed}_{stamp}.csv"
    provenance = provenance or collect_run_provenance()
    if run_id is None:
        grid = sorted({(row["num_reads"], row["num_sweeps"], row["scale_multiplier"])
                       for row in rows})
        run_id = stable_run_id("mnist_qubo_validation_tuning", {
            "seed": seed, "epochs": epochs, "validation_fraction": validation_fraction,
            "representation_artifact_id": representation_artifact_id, "grid": grid,
        }, provenance)
    fields = ["run_id", "grid_protocol", "seed", "epochs", "validation_fraction", "representation_artifact_id",
              "n_train", "n_validation", "n_test", "num_reads", "scale_multiplier",
              "num_sweeps", "solver_seed", "runtime_sec", "validation_accuracy", "selected",
              "status", "error", "selected_candidate_runtime_sec", "tuning_total_seconds",
              "git_commit_sha", "git_worktree_dirty", "python_version", "os_platform", "device",
              "package_versions_json"]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"run_id": run_id, "grid_protocol": "fixed_27_configuration_v1",
                             "seed": seed, "epochs": epochs,
                             "validation_fraction": validation_fraction,
                             "representation_artifact_id": representation_artifact_id,
                             "n_train": "" if n_train is None else n_train,
                             "n_validation": "" if n_validation is None else n_validation,
                             "n_test": "" if n_test is None else n_test,
                             "git_commit_sha": provenance["git_commit_sha"],
                             "git_worktree_dirty": provenance["git_worktree_dirty"],
                             "python_version": provenance["python_version"],
                             "os_platform": provenance["os_platform"],
                             "device": provenance["device"],
                             "package_versions_json": json.dumps(provenance["package_versions"], sort_keys=True),
                             **{key: row.get(key, "") for key in fields[9:20]}})
    return path


def run_tuning(seed=123, epochs=16, num_reads_grid=DEFAULT_NUM_READS,
               num_sweeps_grid=DEFAULT_NUM_SWEEPS,
               scale_multipliers=DEFAULT_SCALE_MULTIPLIERS, validation_fraction=0.15,
               output_dir=Path("data/results_diagnostic"),
               representation_dir=DEFAULT_REPRESENTATION_DIR):
    validate_grid(num_reads_grid, num_sweeps_grid, scale_multipliers)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0.05 <= validation_fraction <= 0.4:
        raise ValueError("validation_fraction must be between 0.05 and 0.4")

    seed = set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    base_train_loader, test_loader = get_mnist_loaders(batch_size=64)
    representation, representation_path, representation_source = load_or_train_mnist_representation(
        base_train_loader.dataset, test_loader.dataset, seed, epochs,
        validation_fraction, device, representation_dir,
    )
    H_train, y_train = representation["H_train"], representation["y_train"]
    H_val, y_val = representation["H_validation"], representation["y_validation"]
    # Test representations and labels are provenance-bound but never scored before selection.
    H_test, y_test = representation["H_test"], representation["y_test"]
    winner, rows = tune_from_representations(H_train, y_train, H_val, y_val, seed,
                                             num_reads_grid, num_sweeps_grid, scale_multipliers)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = representation["metadata"]["sample_counts"]
    provenance = collect_run_provenance(str(device))
    run_id = stable_run_id("mnist_qubo_validation_tuning", {
        "seed": seed, "epochs": epochs, "validation_fraction": validation_fraction,
        "representation_artifact_id": representation["artifact_id"],
        "grid": validate_grid(num_reads_grid, num_sweeps_grid, scale_multipliers),
    }, provenance)
    csv_path = write_tuning_csv(
        rows, seed, epochs, validation_fraction, output_dir,
        representation["artifact_id"], counts["train"], counts["validation"], counts["test"],
        run_id=run_id, provenance=provenance,
    )
    if winner is None:
        failure_summary = {
            "run_id": run_id, "candidate_artifact": csv_path.stem, "seed": seed,
            "epochs": epochs, "validation_fraction": validation_fraction,
            "representation_artifact_id": representation["artifact_id"],
            "grid_protocol": "fixed_27_configuration_v1", "status": "all_candidates_failed",
            "candidate_csv": csv_path.name, "tuning_total_seconds": max(
                (float(row.get("tuning_total_seconds", 0.0)) for row in rows), default=0.0
            ), "provenance": provenance,
        }
        json_path = output_dir / f"{csv_path.stem}.json"
        with json_path.open("x", encoding="utf-8") as handle:
            json.dump(failure_summary, handle, indent=2)
        raise RuntimeError(f"All MNIST QUBO tuning candidates failed; artifacts: {csv_path}, {json_path}")

    # Test set is evaluated exactly once, after selection and candidate artifact persistence.
    test_scores = H_test.float() @ (winner["_weights"] * winner["_scales"].unsqueeze(0)) + winner["_bias"]
    test_accuracy = _accuracy(test_scores, y_test)
    # Match the CSV stem in the optional JSON summary for this run.
    stem = csv_path.stem
    json_path = output_dir / f"{stem}.json"
    summary = {
        "run_id": run_id, "candidate_artifact": stem, "seed": seed, "epochs": epochs,
        "validation_fraction": validation_fraction,
        "representation_artifact_id": representation["artifact_id"],
        "representation_path": str(representation_path.resolve()),
        "representation_cache_status": representation_source,
        "sample_counts": counts,
        "split_protocol": representation["metadata"]["split_protocol"],
        "grid_protocol": "fixed_27_configuration_v1",
        "num_reads_grid": list(num_reads_grid), "num_sweeps_grid": list(num_sweeps_grid),
        "scale_multipliers": list(scale_multipliers),
        "selected_configuration": {key: winner[key] for key in
                                    ("num_reads", "num_sweeps", "scale_multiplier", "solver_seed")},
        "selection_metric": "validation_accuracy_percent",
        "selection_value": winner["validation_accuracy"],
        "test_accuracy_percent": test_accuracy,
        "tuning_total_seconds": winner["tuning_total_seconds"],
        "selected_candidate_runtime_sec": winner["runtime_sec"],
        "provenance": provenance,
        "solver_seed_rule": "(backbone_seed + 1009 * (grid_index + 1)) mod 2**32",
        "tie_break_order": ["highest validation accuracy", "fewest num_reads",
                            "scale_multiplier closest to 1.0", "fewest num_sweeps",
                            "lowest scale_multiplier"],
        "scale_initialization": "Ridge-derived scales with 1e-8 floor, multiplied by candidate multiplier",
        "target_encoding": "one-vs-rest {-1,+1}",
        "final_scale_bias_fit": "training representations and labels only",
    }
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Selected from 27 configurations: num_reads={winner['num_reads']}, "
          f"num_sweeps={winner['num_sweeps']}, scale_multiplier={winner['scale_multiplier']:g}; "
          f"validation={winner['validation_accuracy']:.2f}%; test={test_accuracy:.2f}%")
    print(f"CSV: {csv_path.resolve()}\nJSON: {json_path.resolve()}")
    return summary, csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--num-reads-grid", type=lambda s: _parse_grid(s, int, "num-reads-grid"),
                        default=DEFAULT_NUM_READS, help="comma-separated positive integers")
    parser.add_argument("--num-sweeps-grid", type=lambda s: _parse_grid(s, int, "num-sweeps-grid"),
                        default=DEFAULT_NUM_SWEEPS, help="comma-separated positive integers")
    parser.add_argument("--scale-multipliers", type=lambda s: _parse_grid(s, float, "scale-multipliers"),
                        default=DEFAULT_SCALE_MULTIPLIERS, help="comma-separated positive numbers")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--representation-dir", type=Path, default=DEFAULT_REPRESENTATION_DIR)
    args = parser.parse_args()
    try:
        run_tuning(seed=args.seed, epochs=args.epochs, num_reads_grid=args.num_reads_grid,
                   num_sweeps_grid=args.num_sweeps_grid,
                   scale_multipliers=args.scale_multipliers,
                   validation_fraction=args.validation_fraction,
                   representation_dir=args.representation_dir)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
