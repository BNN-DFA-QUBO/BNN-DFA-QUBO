"""SHAP + LIME explanation experiment: stroke / binarized_ls.

Run from repository root: python -m xai.experiments.stroke.xai_binarized_ls --seed 42
"""
from xai.core.cli import run_cli

if __name__ == "__main__":
    run_cli("stroke", "binarized_ls")
