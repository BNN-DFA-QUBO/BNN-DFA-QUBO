"""SHAP + LIME explanation experiment: stroke / direct_binary.

Run from repository root: python -m xai.experiments.stroke.xai_direct_binary --seed 42
"""
from xai.core.cli import run_cli

if __name__ == "__main__":
    run_cli("stroke", "direct_binary")
