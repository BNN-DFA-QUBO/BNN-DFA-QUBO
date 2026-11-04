"""SHAP + LIME explanation experiment: cifar10 / ste_binary.

Run from repository root: python -m xai.experiments.cifar10.xai_ste_binary --seed 42
"""
from xai.core.cli import run_cli

if __name__ == "__main__":
    run_cli("cifar10", "ste_binary")
