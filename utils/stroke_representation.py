import torch
from pathlib import Path
import hashlib
import json

from utils.stroke_config import PREPROCESSING_VERSION
from utils.stroke_data import METADATA_PATH, RAW_PATH

REPRESENTATION_VERSION = "stroke-representation-v2"


def save_representation(result, path, feature_names=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    names = list(feature_names or result.get("feature_names", []))
    if not names:
        raise ValueError("Cannot save a stroke representation without its fitted feature names")
    metadata = result.get("data_metadata")
    if not metadata:
        raise ValueError("Cannot save a stroke representation without preprocessing provenance")
    if metadata.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("Cannot save a representation with an incompatible preprocessing version")
    data = {
        "representation_version": REPRESENTATION_VERSION,
        "seed": result["seed"],
        "training_seed": result.get("training_seed", result["seed"]),
        "device": result["device"],
        "best_epoch": result["best_epoch"],
        "best_val_ba": result["best_val_ba"],

        "input_size": len(names),
        "hidden_size": int(result["H_train"].shape[1]),
        "split_seed": metadata.get("split_seed"),
        "preprocessing_version": metadata.get("preprocessing_version"),
        "dataset_sha256": metadata.get("dataset_sha256"),
        "balancing_method": metadata.get("balancing_method"),
        "training_config": result.get("training_config", {}),

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

    data["feature_names"] = names
    data["data_metadata"] = metadata

    if "dfa_feedback" in result:
        data["dfa_feedback"] = result["dfa_feedback"]

    torch.save(data, path)


def load_representation(path, expected_metadata=None):
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )
    if data.get("representation_version") != REPRESENTATION_VERSION:
        raise ValueError("Unsupported or stale stroke representation; regenerate it with the current preprocessing pipeline")
    metadata = data.get("data_metadata")
    if not isinstance(metadata, dict) or metadata.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("Representation is missing compatible preprocessing provenance")
    if data.get("preprocessing_version") != metadata.get("preprocessing_version"):
        raise ValueError("Representation preprocessing version disagrees with its embedded metadata")
    if metadata.get("dataset_sha256") != data.get("dataset_sha256") or metadata.get("split_seed") != data.get("split_seed"):
        raise ValueError("Representation dataset hash or split seed disagrees with its metadata")
    representation_metadata_path = Path(path).parent / METADATA_PATH.name
    current_metadata_path = representation_metadata_path if representation_metadata_path.is_file() else METADATA_PATH
    if expected_metadata is None and current_metadata_path.is_file():
        current = json.loads(current_metadata_path.read_text(encoding="utf-8"))
        if current.get("dataset_sha256") != data.get("dataset_sha256"):
            raise ValueError("Representation was created from a different raw dataset; regenerate preprocessing and representation")
        if current.get("preprocessing_config_hash") != metadata.get("preprocessing_config_hash"):
            raise ValueError("Representation preprocessing configuration is stale; regenerate the representation")
        if RAW_PATH.is_file():
            if hashlib.sha256(RAW_PATH.read_bytes()).hexdigest() != data.get("dataset_sha256"):
                raise ValueError("Raw CSV has changed since this representation was generated")
    if expected_metadata is not None:
        for key in ("dataset_sha256", "split_seed", "preprocessing_version", "preprocessing_config_hash"):
            if metadata.get(key) != expected_metadata.get(key):
                raise ValueError(f"Representation {key} does not match the requested experiment")
    for split in ("train", "val", "test"):
        hidden = data[f"H_{split}"]
        labels = data[f"y_{split}"]
        if hidden.shape[0] != labels.shape[0]:
            raise ValueError(f"Representation H_{split} rows do not align with y_{split}")
        if not torch.isfinite(hidden).all():
            raise ValueError(f"Representation H_{split} contains non-finite values")
        if not torch.isin(labels, torch.tensor([0.0, 1.0])).all():
            raise ValueError(f"Representation y_{split} contains non-binary labels")
    names = data.get("feature_names", [])
    if names and len(names) != data.get("input_size"):
        raise ValueError("Representation feature names do not match its saved input dimension")
    if names and metadata.get("feature_names") != names:
        raise ValueError("Representation schema differs from preprocessing metadata")
    if metadata.get("input_size") != data.get("input_size"):
        raise ValueError("Representation input size differs from preprocessing metadata")
    if data.get("hidden_size") != data["H_train"].shape[1]:
        raise ValueError("Representation hidden size does not match its saved arrays")
    split_ids = metadata.get("split_row_ids", {})
    train_ids = split_ids.get("train", [])
    val_ids = split_ids.get("validation", [])
    test_ids = split_ids.get("test", [])
    resampled_ids = metadata.get("resampled_train_row_ids", [])
    if not all((train_ids, val_ids, test_ids, resampled_ids)):
        raise ValueError("Representation metadata is missing split row identities")
    if len(set(train_ids) & set(val_ids) | set(train_ids) & set(test_ids) | set(val_ids) & set(test_ids)):
        raise ValueError("Representation split row identities overlap")
    if not set(resampled_ids) <= set(train_ids):
        raise ValueError("Resampled training rows include identities outside the original training split")
    for split, ids in (("train", resampled_ids), ("val", val_ids), ("test", test_ids)):
        if len(ids) != len(data[f"y_{split}"]):
            raise ValueError(f"Representation y_{split} does not align with saved row identities")
    for split, metadata_key in (("train", "resampled_train_class_counts"),
                                ("val", "validation_class_counts"), ("test", "test_class_counts")):
        labels = data[f"y_{split}"].long()
        counts = {str(int(label)): int((labels == label).sum()) for label in torch.unique(labels)}
        expected_counts = metadata.get(metadata_key, {})
        if counts != expected_counts:
            raise ValueError(f"Representation y_{split} class counts disagree with preprocessing metadata")
    return data
