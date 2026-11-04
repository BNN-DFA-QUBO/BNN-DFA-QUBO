"""SHAP + LIME explanation experiment: mnist / ste_binary.

Run from repository root: python -m xai.experiments.mnist.xai_ste_binary --seed 42
"""
from xai.core.cli import run_cli

if __name__ == "__main__":
    run_cli("mnist", "ste_binary")
