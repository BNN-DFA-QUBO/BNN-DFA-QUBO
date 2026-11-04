"""SHAP + LIME explanation experiment: stroke / real_ls.

Run from repository root: python -m xai.experiments.stroke.xai_real_ls --seed 42
"""
from xai.core.cli import run_cli

if __name__ == "__main__":
    run_cli("stroke", "real_ls")
