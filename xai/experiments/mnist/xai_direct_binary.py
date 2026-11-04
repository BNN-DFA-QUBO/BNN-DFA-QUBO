"""SHAP + LIME explanation experiment: mnist / direct_binary.

Run from repository root: python -m xai.experiments.mnist.xai_direct_binary --seed 42
"""
from xai.core.cli import run_cli

if __name__ == "__main__":
    run_cli("mnist", "direct_binary")
