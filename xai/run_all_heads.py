"""Run XAI experiments on shared classifier-head representations."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from xai.config import XAIConfig
from xai.core.runner import run_xai_experiment


DATASETS = (
    "bank",
    "mnist",
    "cifar10",
    "stroke",
    "resnet18_cifar10",
)

METHODS = (
    "real_ls",
    "binarized_ls",
    "direct_binary",
    "ste_binary",
    "qubo",
)


def check_dependencies(
    include_qubo: bool,
    require_official_lime: bool,
):
    required = [
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "shap",
        "torchvision",
    ]

    if require_official_lime:
        required.append("lime")

    if include_qubo:
        required.append("neal")

    missing = [
        name
        for name in required
        if importlib.util.find_spec(name) is None
    ]

    if missing:
        raise SystemExit(
            "Missing required dependencies: "
            + ", ".join(missing)
            + "\nInstall project and XAI requirements."
        )


def main():

    parser = argparse.ArgumentParser(
        description="Run SHAP + LIME for classifier-head experiments."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=["bank"],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--background-size", type=int, default=40)
    parser.add_argument("--explain-samples", type=int, default=20)
    parser.add_argument("--shap-nsamples", type=int, default=None)
    parser.add_argument("--lime-samples", type=int, default=3000)
    parser.add_argument("--stability-repeats", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-reads", type=int, default=50)
    parser.add_argument("--num-sweeps", type=int, default=500)
    parser.add_argument("--device", default=None)

    parser.add_argument(
        "--allow-lime-fallback",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite-representation",
        action="store_true",
    )

    args = parser.parse_args()

    seeds = (
        args.seeds
        if args.seeds is not None
        else [args.seed]
    )

    if len(set(seeds)) != len(seeds):
        parser.error("Seeds must be unique.")

    if args.explain_samples < 2:
        parser.error("Use at least two explanation samples.")

    if args.background_size < 2:
        parser.error("background-size must be >= 2.")

    if args.stability_repeats < 2:
        parser.error("stability-repeats must be >= 2.")

    if args.lime_samples < 1000:
        parser.error("lime-samples must be >= 1000.")

    check_dependencies(
        include_qubo="qubo" in args.methods,
        require_official_lime=not args.allow_lime_fallback,
    )

    root = Path.cwd()

    if not (root / "xai" / "core" / "runner.py").exists():
        raise SystemExit(
            "Run this command from the BNN-DFA-QUBO repository root."
        )

    total = (
        len(args.datasets)
        * len(args.methods)
        * len(seeds)
    )

    completed = 0

    for seed in seeds:

        for dataset in args.datasets:

            for method in args.methods:

                completed += 1

                print(
                    f"\n[{completed}/{total}] "
                    f"{dataset} / {method} / seed {seed}",
                    flush=True,
                )

                if dataset == "resnet18_cifar10":

                    config = XAIConfig(
                        seed=seed,
                        backbone_epochs=args.epochs,
                        background_size=args.background_size,
                        explain_samples=args.explain_samples,
                        shap_nsamples=args.shap_nsamples,
                        lime_num_samples=args.lime_samples,
                        stability_repeats=args.stability_repeats,
                        top_k=args.top_k,
                        device=args.device,
                        num_reads=args.num_reads,
                        num_sweeps=args.num_sweeps,
                        overwrite_representation=args.overwrite_representation,
                    )

                    # No conversion to ordinary CIFAR-10.
                    run_xai_experiment(
                        "resnet18_cifar10",
                        method,
                        config,
                    )

                else:

                    module = (
                        f"xai.experiments.{dataset}.xai_{method}"
                    )

                    cmd = [
                        sys.executable,
                        "-m",
                        module,
                        "--seed",
                        str(seed),
                        "--background-size",
                        str(args.background_size),
                        "--explain-samples",
                        str(args.explain_samples),
                        "--lime-samples",
                        str(args.lime_samples),
                        "--stability-repeats",
                        str(args.stability_repeats),
                        "--top-k",
                        str(args.top_k),
                        "--num-reads",
                        str(args.num_reads),
                        "--num-sweeps",
                        str(args.num_sweeps),
                    ]

                    if args.epochs is not None:
                        cmd += ["--epochs", str(args.epochs)]

                    if args.shap_nsamples is not None:
                        cmd += [
                            "--shap-nsamples",
                            str(args.shap_nsamples),
                        ]

                    if args.device:
                        cmd += ["--device", args.device]

                    if args.overwrite_representation:
                        cmd.append("--overwrite-representation")

                    subprocess.run(cmd, check=True)

    print(
        f"\nCompleted all {total} requested XAI runs.",
        flush=True,
    )

    print(
        "\nCollect results using:\n"
        "python -m xai.collect_xai_results "
        "--datasets "
        + " ".join(args.datasets)
        + " --seeds "
        + " ".join(map(str, seeds)),
        flush=True,
    )


if __name__ == "__main__":
    main()