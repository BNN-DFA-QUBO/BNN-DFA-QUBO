"""CLI helper used by deliberately separate hard-coded dataset/method files."""
from __future__ import annotations
import argparse
from pathlib import Path
from xai.config import XAIConfig
from xai.core.runner import run_xai_experiment


def run_cli(dataset: str, method: str):
    parser = argparse.ArgumentParser(description=f"Run SHAP + LIME for {dataset} / {method} on the repository's shared BNN-DFA representation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Backbone epochs (dataset default when omitted)")
    parser.add_argument("--background-size", type=int, default=40)
    parser.add_argument("--explain-samples", type=int, default=8)
    parser.add_argument("--shap-nsamples", type=int, default=None)
    parser.add_argument("--lime-samples", type=int, default=2000)
    parser.add_argument("--stability-repeats", type=int, default=2,
                        help="Independent SHAP background/LIME perturbation repeats per explained row")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("data/xai_results"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("data/xai_artifacts"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--overwrite-representation", action="store_true")
    args = parser.parse_args()
    config = XAIConfig(
        seed=args.seed, backbone_epochs=args.epochs, background_size=args.background_size,
        explain_samples=args.explain_samples, shap_nsamples=args.shap_nsamples,
        lime_num_samples=args.lime_samples, stability_repeats=args.stability_repeats,
        top_k=args.top_k,
        output_dir=args.output_dir, artifact_dir=args.artifact_dir, device=args.device,
        num_reads=args.num_reads, num_sweeps=args.num_sweeps,
        overwrite_representation=args.overwrite_representation,
    )
    run_xai_experiment(dataset, method, config)
