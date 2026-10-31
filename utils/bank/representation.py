"""Validated save/load helpers for fixed Bank BNN+DFA representations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from utils.run_provenance import collect_run_provenance

REPRESENTATION_VERSION = "bank-bnn-dfa-representation-v1"
DEFAULT_REPRESENTATION_DIR = Path("data/bank_representations")

PREPROCESSING = {
    "feature_count": 66,
    "yeo_johnson": True,
    "knn_imputed": True,
    "one_hot_encoded": True,
}

BACKBONE = {
    "class": "models.bank_bnn.BankBNN",
    "fc1": {"in_features": 66, "out_features": 64, "activation": "relu"},
    "fc2": {"in_features": 64, "out_features": 2},
    "training_method": "BankDFA",
}


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def dataset_fingerprint(train_dataset, val_dataset, test_dataset) -> str:
    """Compute SHA-256 fingerprint over train, val, and test data and targets."""
    digest = hashlib.sha256()
    for name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        digest.update(name.encode("ascii"))
        for attribute in ("data", "targets"):
            array = np.ascontiguousarray(_numpy(getattr(dataset, attribute)))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def build_metadata(
    train_dataset,
    val_dataset,
    test_dataset,
    seed: int,
    epochs: int,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    hidden_size: int = 64,
    device: str = "cpu",
) -> dict[str, Any]:
    """Build provenance metadata for bank representations."""
    return {
        "version": REPRESENTATION_VERSION,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "hidden_size": int(hidden_size),
        "device": str(device),
        "preprocessing": PREPROCESSING,
        "backbone": BACKBONE,
        "dataset_fingerprint": dataset_fingerprint(train_dataset, val_dataset, test_dataset),
        "provenance": collect_run_provenance(),
    }


def representation_path(output_dir: Path | str, seed: int, epochs: int) -> Path:
    output_dir = Path(output_dir)
    return output_dir / f"bank_repr_seed_{seed}_epochs_{epochs}.pt"


def metadata_path(repr_path: Path) -> Path:
    return repr_path.with_suffix(".json")


def save_representation(
    path: Path | str,
    payload: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> Path:
    """Save representations alongside JSON metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate payload structure
    required_keys = {"H_train", "y_train", "H_val", "y_val", "H_test", "y_test"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"Payload missing required keys: {missing}")

    torch.save(payload, path)

    m_path = metadata_path(path)
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return path


def validate_representation(payload: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    """Verify integrity, finite values, and dimensions of representation payload."""
    required_keys = ["H_train", "y_train", "H_val", "y_val", "H_test", "y_test"]
    for k in required_keys:
        if k not in payload:
            raise KeyError(f"Missing {k} in representation payload")
        t = payload[k]
        if not torch.is_tensor(t):
            raise TypeError(f"Expected torch.Tensor for {k}, got {type(t)}")
        if not torch.isfinite(t.float()).all():
            raise ValueError(f"Non-finite values found in {k}")

    hidden_size = metadata.get("hidden_size", 64)
    if payload["H_train"].shape[1] != hidden_size:
        raise ValueError(f"H_train hidden size {payload['H_train'].shape[1]} != {hidden_size}")
    if payload["H_train"].shape[0] != payload["y_train"].shape[0]:
        raise ValueError("H_train and y_train row counts do not match")
    if payload["H_val"].shape[0] != payload["y_val"].shape[0]:
        raise ValueError("H_val and y_val row counts do not match")
    if payload["H_test"].shape[0] != payload["y_test"].shape[0]:
        raise ValueError("H_test and y_test row counts do not match")


def load_representation(
    path: Path | str,
    expected_metadata: dict[str, Any],
    train_dataset: Any = None,
    val_dataset: Any = None,
    test_dataset: Any = None,
) -> dict[str, torch.Tensor]:
    """Load and validate cached representations."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Representation file not found: {path}")

    m_path = metadata_path(path)
    if not m_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {m_path}")

    with open(m_path, "r", encoding="utf-8") as f:
        cached_meta = json.load(f)

    # Check key invariants
    for check_key in ("version", "seed", "epochs", "hidden_size"):
        if check_key in expected_metadata:
            if cached_meta.get(check_key) != expected_metadata[check_key]:
                raise ValueError(
                    f"Metadata mismatch on {check_key}: cached {cached_meta.get(check_key)} != expected {expected_metadata[check_key]}"
                )

    if train_dataset is not None and val_dataset is not None and test_dataset is not None:
        current_fp = dataset_fingerprint(train_dataset, val_dataset, test_dataset)
        cached_fp = cached_meta.get("dataset_fingerprint")
        if cached_fp and cached_fp != current_fp:
            raise ValueError(f"Dataset fingerprint mismatch: cached {cached_fp} != current {current_fp}")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_representation(payload, cached_meta)
    return payload

