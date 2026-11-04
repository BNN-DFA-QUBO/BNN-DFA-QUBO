"""Run the five XAI head comparisons with a consistent, explicit protocol.

Run from repository root, e.g.:
python -m xai.run_all_heads --datasets bank --seed 42
python -m xai.run_all_heads --datasets bank mnist cifar10 stroke --seeds 42 123 2024 7 99
"""
from __future__ import annotations
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

DATASETS = ("bank", "mnist", "cifar10", "stroke", "resnet18_cifar10")
METHODS = ("real_ls", "binarized_ls", "direct_binary", "ste_binary", "qubo")


def check_dependencies(include_qubo: bool, require_official_lime: bool) -> None:
    required = ["numpy", "pandas", "sklearn", "matplotlib", "shap"]
    if require_official_lime:
        required.append("lime")
    if include_qubo:
        required.append("neal")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            "Missing required XAI dependencies: " + ", ".join(missing) +
            "\nInstall with: python -m pip install -r requirements.txt && "
            "python -m pip install -r xai/requirements.txt"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHAP+LIME for the five shared-representation classifier heads.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["bank"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Single seed; ignored when --seeds is supplied")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--background-size", type=int, default=40)
    parser.add_argument("--explain-samples", type=int, default=20)
    parser.add_argument("--shap-nsamples", type=int, default=None)
    parser.add_argument("--lime-samples", type=int, default=3000)
    parser.add_argument("--stability-repeats", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-lime-fallback", action="store_true",
                        help="Permit internal weighted-Ridge fallback; not recommended for final official-LIME evidence")
    parser.add_argument("--overwrite-representation", action="store_true")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(seeds)) != len(seeds):
        parser.error("Seeds must be unique")
    if args.explain_samples < 2:
        parser.error("Use at least 2 explained test rows for meaningful sample-level diagnostics")
    if args.background_size < 2 or args.stability_repeats < 2:
        parser.error("Use background-size >= 2 and stability-repeats >= 2")
    if args.lime_samples < 1000:
        parser.error("Use at least 1000 LIME perturbations for the recommended protocol")

    check_dependencies("qubo" in args.methods, not args.allow_lime_fallback)
    root = Path.cwd()
    if not (root / "xai" / "core" / "runner.py").exists():
        raise SystemExit("Run this command from the BNN-DFA-QUBO repository root.")

    total = len(args.datasets) * len(args.methods) * len(seeds)
    completed = 0
    for seed in seeds:
        for dataset in args.datasets:
            for method in args.methods:
                completed += 1
                module_dataset = "cifar10" if dataset == "resnet18_cifar10" else dataset
                module = f"xai.experiments.{module_dataset}.xai_{method}"
                cmd = [sys.executable, "-m", module, "--seed", str(seed),
                       "--background-size", str(args.background_size),
                       "--explain-samples", str(args.explain_samples),
                       "--lime-samples", str(args.lime_samples),
                       "--stability-repeats", str(args.stability_repeats),
                       "--top-k", str(args.top_k),
                       "--num-reads", str(args.num_reads), "--num-sweeps", str(args.num_sweeps)]
                if args.epochs is not None:
                    cmd += ["--epochs", str(args.epochs)]
                if args.shap_nsamples is not None:
                    cmd += ["--shap-nsamples", str(args.shap_nsamples)]
                if args.device:
                    cmd += ["--device", args.device]
                if args.overwrite_representation:
                    cmd.append("--overwrite-representation")
                print(f"\n[{completed}/{total}] {dataset} / {method} / seed {seed}", flush=True)
                subprocess.run(cmd, check=True)
    print(f"\nCompleted all {total} requested XAI runs.", flush=True)
    print("Next: python -m xai.collect_xai_results --datasets " + " ".join(args.datasets) +
          " --seeds " + " ".join(map(str, seeds)), flush=True)


if __name__ == "__main__":
    main()
