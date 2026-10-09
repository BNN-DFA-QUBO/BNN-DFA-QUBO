import torch
from pathlib import Path

from utils.stroke_config import HIDDEN_SIZE


def save_representation(result, path, feature_names=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "seed": result["seed"],
        "device": result["device"],
        "best_epoch": result["best_epoch"],
        "best_val_ba": result["best_val_ba"],

        "input_size": len(feature_names or result.get("feature_names", [])),
        "hidden_size": HIDDEN_SIZE,
        "split_seed": result.get("data_metadata", {}).get("seed"),

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
    if "data_metadata" in result:
        data["data_metadata"] = result["data_metadata"]

    if "dfa_feedback" in result:
        data["dfa_feedback"] = result["dfa_feedback"]

    torch.save(data, path)


def load_representation(path):
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )
    for split in ("train", "val", "test"):
        hidden = data[f"H_{split}"]
        labels = data[f"y_{split}"]
        if hidden.shape[0] != labels.shape[0]:
            raise ValueError(f"Representation H_{split} rows do not align with y_{split}")
        if not torch.isfinite(hidden).all():
            raise ValueError(f"Representation H_{split} contains non-finite values")
    names = data.get("feature_names", [])
    if names and len(names) != data.get("input_size"):
        raise ValueError("Representation feature names do not match its saved input dimension")
    if "data_metadata" in data and names and data["data_metadata"].get("feature_names") != names:
        raise ValueError("Representation schema differs from preprocessing metadata")
    return data
