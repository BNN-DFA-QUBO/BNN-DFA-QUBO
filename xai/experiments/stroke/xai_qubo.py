"""SHAP + LIME explanation experiment: stroke / qubo.

Run from repository root: python -m xai.experiments.stroke.xai_qubo --seed 42
"""
from xai.core.cli import run_cli

if __name__ == "__main__":
    run_cli("stroke", "qubo")
