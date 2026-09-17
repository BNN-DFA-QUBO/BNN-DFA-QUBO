import torch
from pathlib import Path

from utils.stroke_config import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    SPLIT_SEED
)


def save_representation(result, path, feature_names=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "seed": result["seed"],
        "device": result["device"],
        "best_epoch": result["best_epoch"],
        "best_val_ba": result["best_val_ba"],

        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "split_seed": SPLIT_SEED,

        "H_train": result["H_train"],
        "H_val": result["H_val"],
        "H_test": result["H_test"],

        "y_train": torch.tensor(
            result["y_train"].values,
            dtype=torch.float32
        ),

        "y_val": torch.tensor(
            result["y_val"].values,
            dtype=torch.float32
        ),

        "y_test": torch.tensor(
            result["y_test"].values,
            dtype=torch.float32
        )
    }

    if feature_names is not None:
        data["feature_names"] = list(feature_names)

    if "dfa_feedback" in result:
        data["dfa_feedback"] = result["dfa_feedback"]

    torch.save(data, path)


def load_representation(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )