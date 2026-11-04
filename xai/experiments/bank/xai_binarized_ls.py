"""SHAP + LIME explanation experiment: bank / binarized_ls.

Run from repository root: python -m xai.experiments.bank.xai_binarized_ls --seed 42
"""
from xai.core.cli import run_cli

if __name__ == "__main__":
    run_cli("bank", "binarized_ls")
