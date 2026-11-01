"""Validated save/load helpers for fixed CIFAR-10 BNN+DFA representations.

Follows the verified MNIST representation protocol with:
1. Deterministic stratified 45,000 / 5,000 / 10,000 split.
2. Dataset fingerprinting based on raw image bytes and targets.
3. Strict SHA-256 tensor digest and artifact identifier binding.
4. Comprehensive metadata validation (architecture, training config, split indices).
5. Overwrite protection with exclusive file creation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from utils.run_provenance import collect_run_provenance

REPRESENTATION_VERSION = "cifar10-bnn-dfa-representation-v1"
DEFAULT_REPRESENTATION_DIR = Path("data/cifar10_representations")
PREPROCESSING = {
    "transform": (
        "torchvision.ToTensor + Normalize("
        "mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))"
    ),
    "input_shape": [3, 32, 32],
    "color_space": "RGB",
}
BACKBONE = {
    "class": "models.cifar_bnn.CIFAR_BNN",
    "conv1": {
        "in_channels": 3,
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "activation": "relu",
    },
    "conv2": {
        "in_channels": 32,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "activation": "relu",
    },
    "gap": {"type": "AdaptiveAvgPool2d((1, 1))", "out_channels": 64},
    "fc1": {"in_features": 64, "out_features": 128, "activation": "relu"},
    "fc2": {"in_features": 128, "out_features": 10},
    "training_method": "CIFARDFA",
}
# Parameters:
# conv1: 32 * 3 * 3 * 3 + 32 = 896
# conv2: 64 * 32 * 3 * 3 + 64 = 18496
# fc1: 128 * 64 + 128 = 8320
# fc2: 10 * 128 + 10 = 1290
# Total: 29,002
BACKBONE_PARAMETER_COUNT = 896 + 18496 + 8320 + 1290


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def dataset_fingerprint(train_dataset: Any, test_dataset: Any) -> str:
    """Compute SHA-256 fingerprint over raw training and testing images and labels."""
    digest = hashlib.sha256()
    for name, dataset in (("train", train_dataset), ("test", test_dataset)):
        digest.update(name.encode("ascii"))
        for attribute in ("data", "targets"):
            array = np.ascontiguousarray(_numpy(getattr(dataset, attribute)))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def split_indices(
    train_dataset: Any,
    test_dataset: Any,
    seed: int,
    validation_fraction: float,
) -> Dict[str, list[int]]:
    """Compute deterministic stratified indices for training, validation, and testing."""
    train_labels = _numpy(train_dataset.targets).reshape(-1)
    train, validation = train_test_split(
        np.arange(len(train_dataset)),
        test_size=float(validation_fraction),
        random_state=int(seed),
        stratify=train_labels,
    )
    return {
        "train": train.astype(np.int64).tolist(),
        "validation": validation.astype(np.int64).tolist(),
        "test": np.arange(len(test_dataset), dtype=np.int64).tolist(),
    }


def build_metadata(
    train_dataset: Any,
    test_dataset: Any,
    seed: int,
    epochs: int,
    validation_fraction: float,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Construct complete metadata structure bound to the representation artifact."""
    indices = split_indices(train_dataset, test_dataset, seed, validation_fraction)
    train_labels = _numpy(train_dataset.targets).reshape(-1)
    test_labels = _numpy(test_dataset.targets).reshape(-1)

    return {
        "representation_version": REPRESENTATION_VERSION,
        "dataset": "CIFAR-10",
        "dataset_fingerprint": dataset_fingerprint(train_dataset, test_dataset),
        "split_protocol": "official_train_stratified_train_validation_holdout_v1",
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "split_indices": indices,
        "sample_counts": {
            "train": len(indices["train"]),
            "validation": len(indices["validation"]),
            "test": len(indices["test"]),
        },
        "class_counts": {
            "train": {
                str(int(k)): int(v)
                for k, v in zip(
                    *np.unique(train_labels[indices["train"]], return_counts=True)
                )
            },
            "validation": {
                str(int(k)): int(v)
                for k, v in zip(
                    *np.unique(train_labels[indices["validation"]], return_counts=True)
                )
            },
            "test": {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(test_labels, return_counts=True))
            },
        },
        "backbone": BACKBONE,
        "backbone_parameter_count": BACKBONE_PARAMETER_COUNT,
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "optimizer": "Adam",
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "dfa_hidden_size": 128,
            "classes": 10,
            "device": str(device),
        },
        "preprocessing": PREPROCESSING,
        "representation_layer": "relu(fc1(gap(relu(conv2(relu(conv1(image)))))))",
        "representation_dim": 128,
        "representation_dtype": "float32",
        "run_provenance": collect_run_provenance(str(device)),
    }


def _tensor_digest(data: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in (
        "H_train",
        "y_train",
        "H_validation",
        "y_validation",
        "H_test",
        "y_test",
    ):
        tensor = data[key].detach().cpu().contiguous()
        digest.update(key.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _compatibility_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    compatible = dict(metadata)
    compatible.pop("run_provenance", None)
    training = compatible.get("training")
    if isinstance(training, dict):
        compatible["training"] = dict(training)
        compatible["training"].pop("device", None)
    return compatible


def _metadata_mismatch_fields(actual: Any, expected: Any, prefix: str = "") -> list[str]:
    if isinstance(actual, dict) and isinstance(expected, dict):
        fields = []
        for key in sorted(set(actual) | set(expected)):
            child = f"{prefix}.{key}" if prefix else key
            if key not in actual or key not in expected:
                fields.append(child)
            else:
                fields.extend(
                    _metadata_mismatch_fields(actual[key], expected[key], child)
                )
        return fields
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) == len(expected):
            fields = []
            for index, (left, right) in enumerate(zip(actual, expected)):
                fields.extend(
                    _metadata_mismatch_fields(left, right, f"{prefix}[{index}]")
                )
            return fields
    return [] if actual == expected else [prefix]


def representation_path(
    output_dir: Path | str,
    seed: int,
    epochs: int,
    validation_fraction: float,
) -> Path:
    fraction = f"{float(validation_fraction):.4f}".replace(".", "p")
    return (
        Path(output_dir)
        / f"cifar10_bnn_dfa_seed_{seed}_epochs_{epochs}_val_{fraction}.pt"
    )


def save_representation(
    path: Path | str,
    data: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Atomically save representation dictionary with strict integrity hashes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite CIFAR-10 representation artifact: {path}"
        )
    saved = dict(data)
    saved["metadata"] = metadata
    saved["representation_sha256"] = _tensor_digest(saved)
    metadata_bytes = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    saved["artifact_id"] = hashlib.sha256(
        metadata_bytes + saved["representation_sha256"].encode("ascii")
    ).hexdigest()

    created = False
    try:
        with path.open("xb") as handle:
            created = True
            torch.save(saved, handle)
    except Exception:
        if created and path.exists():
            path.unlink()
        raise
    return saved


def load_representation(
    path: Path | str,
    expected_metadata: Dict[str, Any],
    train_dataset: Any,
    test_dataset: Any,
) -> Dict[str, Any]:
    """Load and strictly validate a saved CIFAR-10 representation artifact."""
    path = Path(path)
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(
            f"CIFAR-10 representation artifact could not be deserialized: {path}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            "CIFAR-10 representation artifact is malformed: expected a dictionary"
        )
    actual_metadata = data.get("metadata")
    if not isinstance(actual_metadata, dict):
        raise ValueError(
            "CIFAR-10 representation artifact is malformed: metadata dictionary missing"
        )
    mismatch_fields = _metadata_mismatch_fields(
        _compatibility_metadata(actual_metadata),
        _compatibility_metadata(expected_metadata),
    )
    if mismatch_fields:
        details = ", ".join(mismatch_fields[:12])
        if len(mismatch_fields) > 12:
            details += f", and {len(mismatch_fields) - 12} more"
        raise ValueError(f"CIFAR-10 representation metadata mismatch in: {details}")

    try:
        actual_tensor_digest = _tensor_digest(data)
    except (KeyError, AttributeError, TypeError, RuntimeError) as exc:
        raise ValueError(
            "CIFAR-10 representation artifact is malformed: required tensors missing/invalid"
        ) from exc
    if data.get("representation_sha256") != actual_tensor_digest:
        raise ValueError("CIFAR-10 representation artifact tensor checksum is invalid")

    metadata_bytes = json.dumps(
        actual_metadata, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    expected_id = hashlib.sha256(
        metadata_bytes + data["representation_sha256"].encode("ascii")
    ).hexdigest()
    if data.get("artifact_id") != expected_id:
        raise ValueError("CIFAR-10 representation artifact identifier is invalid")

    indices = expected_metadata["split_indices"]
    train_y = torch.as_tensor(
        _numpy(train_dataset.targets)[indices["train"]], dtype=torch.long
    )
    val_y = torch.as_tensor(
        _numpy(train_dataset.targets)[indices["validation"]], dtype=torch.long
    )
    test_y = torch.as_tensor(
        _numpy(test_dataset.targets)[indices["test"]], dtype=torch.long
    )

    for name, expected_labels in (
        ("train", train_y),
        ("validation", val_y),
        ("test", test_y),
    ):
        features, labels = data[f"H_{name}"], data[f"y_{name}"]
        if features.ndim != 2 or features.shape != (len(expected_labels), 128):
            raise ValueError(
                f"CIFAR-10 representation H_{name} has incompatible shape: {features.shape}"
            )
        if features.dtype != torch.float32 or labels.dtype != torch.int64:
            raise ValueError(
                f"CIFAR-10 representation H_{name}/y_{name} has incompatible dtype"
            )
        if labels.shape != expected_labels.shape or not torch.equal(
            labels.long(), expected_labels
        ):
            raise ValueError(
                f"CIFAR-10 representation y_{name} misaligned with saved indices"
            )
        if not torch.isfinite(features).all():
            raise ValueError(
                f"CIFAR-10 representation H_{name} contains non-finite values"
            )

    return data
