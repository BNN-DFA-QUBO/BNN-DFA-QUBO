"""SHAP + LIME explanation experiment: mnist / real_ls.

Run from repository root: python -m xai.experiments.mnist.xai_real_ls --seed 42
"""
from xai.core.cli import run_cli

if __name__ == "__main__":
    run_cli("mnist", "real_ls")
