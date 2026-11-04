"""Configuration and stable names for the XAI experiment layer."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

DATASETS = ("mnist", "stroke", "bank", "cifar10")
METHODS = ("real_ls", "binarized_ls", "direct_binary", "ste_binary", "qubo")
DEFAULT_OUTPUT_DIR = Path("data/xai_results")
DEFAULT_ARTIFACT_DIR = Path("data/xai_artifacts")

@dataclass(frozen=True)
class XAIConfig:
    seed: int = 42
    backbone_epochs: int | None = None
    background_size: int = 40
    explain_samples: int = 8
    shap_nsamples: int | None = None  # None -> 2 * n_features + 2048
    lime_num_samples: int = 2000
    stability_repeats: int = 2
    random_faithfulness_repeats: int = 20
    top_k: int = 10
    output_dir: Path = DEFAULT_OUTPUT_DIR
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    device: str | None = None
    num_reads: int = 100
    num_sweeps: int = 1000
    overwrite_representation: bool = False

    def validate(self) -> None:
        if self.background_size < 2 or self.explain_samples < 1:
            raise ValueError("background_size must be >= 2 and explain_samples >= 1")
        if self.lime_num_samples < 100 or self.stability_repeats < 1:
            raise ValueError("lime_num_samples must be >= 100 and stability_repeats >= 1")
        if self.random_faithfulness_repeats < 1 or self.top_k < 1:
            raise ValueError("faithfulness repeats and top_k must be positive")
        if self.shap_nsamples is not None and self.shap_nsamples < 2:
            raise ValueError("shap_nsamples must be >= 2")
        if self.num_reads < 1 or self.num_sweeps < 1:
            raise ValueError("QUBO num_reads and num_sweeps must be positive")
