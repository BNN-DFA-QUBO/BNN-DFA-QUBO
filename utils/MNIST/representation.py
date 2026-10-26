"""Validated save/load helpers for fixed MNIST BNN+DFA representations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

REPRESENTATION_VERSION = "mnist-bnn-dfa-representation-v1"
DEFAULT_REPRESENTATION_DIR = Path("data/mnist_representations")
PREPROCESSING = {
    "transform": "torchvision.ToTensor + Normalize(mean=0.1307,std=0.3081)",
    "input_shape": [1, 28, 28],
    "flatten_order": "row-major",
}
BACKBONE = {
    "class": "models.bnn.BNN",
    "fc1": {"in_features": 784, "out_features": 128, "activation": "relu"},
    "fc2": {"in_features": 128, "out_features": 10},
    "training_method": "DFAClassifier",
}


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def dataset_fingerprint(train_dataset, test_dataset):
    digest = hashlib.sha256()
    for name, dataset in (("train", train_dataset), ("test", test_dataset)):
        digest.update(name.encode("ascii"))
        for attribute in ("data", "targets"):
            array = np.ascontiguousarray(_numpy(getattr(dataset, attribute)))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def split_indices(train_dataset, test_dataset, seed, validation_fraction):
    train_labels = _numpy(train_dataset.targets).reshape(-1)
    train, validation = train_test_split(
        np.arange(len(train_dataset)), test_size=validation_fraction,
        random_state=int(seed), stratify=train_labels,
    )
    return {
        "train": train.astype(np.int64).tolist(),
        "validation": validation.astype(np.int64).tolist(),
        "test": np.arange(len(test_dataset), dtype=np.int64).tolist(),
    }


def build_metadata(train_dataset, test_dataset, seed, epochs, validation_fraction,
                   batch_size=64, learning_rate=0.001, device="cpu"):
    indices = split_indices(train_dataset, test_dataset, seed, validation_fraction)
    train_labels = _numpy(train_dataset.targets).reshape(-1)
    test_labels = _numpy(test_dataset.targets).reshape(-1)
    return {
        "representation_version": REPRESENTATION_VERSION,
        "dataset": "MNIST",
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
            "train": {str(int(k)): int(v) for k, v in zip(*np.unique(train_labels[indices["train"]], return_counts=True))},
            "validation": {str(int(k)): int(v) for k, v in zip(*np.unique(train_labels[indices["validation"]], return_counts=True))},
            "test": {str(int(k)): int(v) for k, v in zip(*np.unique(test_labels, return_counts=True))},
        },
        "backbone": BACKBONE,
        "backbone_parameter_count": 28 * 28 * 128 + 128 + 128 * 10 + 10,
        "training": {
            "epochs": int(epochs), "batch_size": int(batch_size),
            "optimizer": "Adam", "learning_rate": float(learning_rate),
            "seed": int(seed), "dfa_hidden_size": 128, "classes": 10,
            "device": str(device),
        },
        "preprocessing": PREPROCESSING,
        "representation_layer": "relu(model.fc1(flatten(normalized_image)))",
        "representation_dim": 128,
        "representation_dtype": "float32",
    }


def _tensor_digest(data):
    digest = hashlib.sha256()
    for key in ("H_train", "y_train", "H_validation", "y_validation", "H_test", "y_test"):
        tensor = data[key].detach().cpu().contiguous()
        digest.update(key.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def representation_path(output_dir, seed, epochs, validation_fraction):
    fraction = f"{float(validation_fraction):.4f}".replace(".", "p")
    return Path(output_dir) / f"mnist_bnn_dfa_seed_{seed}_epochs_{epochs}_val_{fraction}.pt"


def save_representation(path, data, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite MNIST representation artifact: {path}")
    saved = dict(data)
    saved["metadata"] = metadata
    saved["representation_sha256"] = _tensor_digest(saved)
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    saved["artifact_id"] = hashlib.sha256(metadata_bytes + saved["representation_sha256"].encode("ascii")).hexdigest()
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


def load_representation(path, expected_metadata, train_dataset, test_dataset):
    path = Path(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("metadata") != expected_metadata:
        raise ValueError("MNIST representation artifact is incompatible with the requested seed, split, dataset, or backbone configuration")
    if data.get("representation_sha256") != _tensor_digest(data):
        raise ValueError("MNIST representation artifact tensor checksum is invalid")
    metadata_bytes = json.dumps(expected_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected_id = hashlib.sha256(metadata_bytes + data["representation_sha256"].encode("ascii")).hexdigest()
    if data.get("artifact_id") != expected_id:
        raise ValueError("MNIST representation artifact identifier is invalid")

    indices = expected_metadata["split_indices"]
    train_y = torch.as_tensor(_numpy(train_dataset.targets)[indices["train"]], dtype=torch.long)
    val_y = torch.as_tensor(_numpy(train_dataset.targets)[indices["validation"]], dtype=torch.long)
    test_y = torch.as_tensor(_numpy(test_dataset.targets)[indices["test"]], dtype=torch.long)
    for name, expected_labels in (("train", train_y), ("validation", val_y), ("test", test_y)):
        features, labels = data[f"H_{name}"], data[f"y_{name}"]
        if features.ndim != 2 or features.shape != (len(expected_labels), 128):
            raise ValueError(f"MNIST representation H_{name} has an incompatible shape")
        if features.dtype != torch.float32 or labels.dtype != torch.int64:
            raise ValueError(f"MNIST representation H_{name}/y_{name} has an incompatible dtype")
        if labels.shape != expected_labels.shape or not torch.equal(labels.long(), expected_labels):
            raise ValueError(f"MNIST representation y_{name} is misaligned with its saved example indices")
        if not torch.isfinite(features).all():
            raise ValueError(f"MNIST representation H_{name} contains non-finite values")
    return data
