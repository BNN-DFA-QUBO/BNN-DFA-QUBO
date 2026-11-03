"""Validated save/load helpers for fixed CIFAR-10 BNN+DFA representations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from utils.run_provenance import collect_run_provenance


REPRESENTATION_VERSION = "cifar10-bnn-dfa-representation-v4"

DEFAULT_REPRESENTATION_DIR = Path("data/cifar10_representations")

REPRESENTATION_DIM = 256
NUM_CLASSES = 10

PREPROCESSING = {
    "transform": (
        "torchvision.ToTensor + Normalize("
        "mean=(0.4914, 0.4822, 0.4465), "
        "std=(0.2470, 0.2435, 0.2616))"
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
    "gap": {
        "type": "AdaptiveAvgPool2d((4, 4))",
        "out_channels": 1024,
    },
    "fc1": {
        "in_features": 1024,
        "out_features": REPRESENTATION_DIM,
        "activation": "relu",
    },
    "fc2": {
        "in_features": REPRESENTATION_DIM,
        "out_features": NUM_CLASSES,
    },
    "training_method": "CIFARDFA",
}

BACKBONE_PARAMETER_COUNT = 896 + 18496 + (1024 * 256 + 256) + (256 * 10 + 10)

TRAINING_DEFAULTS = {
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "scheduler": {
        "type": "CosineAnnealingLR",
        "eta_min": 5e-6,
    },
    "gradient_clipping": {
        "strategy": "layer_wise",
        "max_norm": 0.25,
    },
    "augmentation": "get_train_transform",
    "feedback_initialization": (
        "normal_std_1_over_sqrt_num_classes"
    ),
    "checkpoint_selection": (
        "lowest_validation_cross_entropy_v1"
    ),
}


def _numpy(value: Any) -> np.ndarray:
    """Convert a tensor or array-like object to NumPy."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def dataset_fingerprint(
    train_dataset: Any,
    test_dataset: Any,
) -> str:
    """Compute SHA-256 fingerprint over raw images and labels."""

    digest = hashlib.sha256()

    for name, dataset in (
        ("train", train_dataset),
        ("test", test_dataset),
    ):
        for attribute in ("data", "targets"):
            if not hasattr(dataset, attribute):
                raise ValueError(
                    f"{name} dataset is missing '{attribute}'."
                )

            array = np.ascontiguousarray(
                _numpy(getattr(dataset, attribute))
            )

            digest.update(name.encode("ascii"))
            digest.update(attribute.encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))

            digest.update(
                np.asarray(
                    array.shape,
                    dtype=np.int64,
                ).tobytes()
            )

            digest.update(array.tobytes())

    return digest.hexdigest()


def split_indices(
    train_dataset: Any,
    test_dataset: Any,
    seed: int,
    validation_fraction: float,
) -> dict[str, list[int]]:
    """Compute deterministic stratified train/validation/test indices."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    train_labels = _numpy(
        train_dataset.targets
    ).reshape(-1)

    if len(train_labels) != len(train_dataset):
        raise ValueError(
            "Training labels do not match dataset length."
        )

    test_labels = _numpy(
        test_dataset.targets
    ).reshape(-1)

    if len(test_labels) != len(test_dataset):
        raise ValueError(
            "Test labels do not match dataset length."
        )

    train, validation = train_test_split(
        np.arange(len(train_dataset)),
        test_size=float(validation_fraction),
        random_state=int(seed),
        stratify=train_labels,
    )

    return {
        "train": train.astype(np.int64).tolist(),
        "validation": validation.astype(np.int64).tolist(),
        "test": np.arange(
            len(test_dataset),
            dtype=np.int64,
        ).tolist(),
    }


def build_metadata(
    train_dataset: Any,
    test_dataset: Any,
    seed: int,
    epochs: int,
    validation_fraction: float,
    batch_size: int,
    learning_rate: float,
    device: str,
    weight_decay: float = TRAINING_DEFAULTS["weight_decay"],
    clipping_strategy: str = TRAINING_DEFAULTS[
        "gradient_clipping"
    ]["strategy"],
    max_grad_norm: float = TRAINING_DEFAULTS[
        "gradient_clipping"
    ]["max_norm"],
) -> dict[str, Any]:
    """Construct metadata bound to the representation artifact."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    if clipping_strategy not in {"layer_wise", "global", "none"}:
        raise ValueError("Invalid clipping strategy.")

    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    indices = split_indices(
        train_dataset,
        test_dataset,
        seed,
        validation_fraction,
    )

    train_labels = _numpy(
        train_dataset.targets
    ).reshape(-1)

    test_labels = _numpy(
        test_dataset.targets
    ).reshape(-1)

    def make_class_counts(
        labels: np.ndarray,
    ) -> dict[str, int]:
        classes, counts = np.unique(
            labels,
            return_counts=True,
        )

        return {
            str(int(label)): int(count)
            for label, count in zip(classes, counts)
        }

    training = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": TRAINING_DEFAULTS["optimizer"],
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "dfa_hidden_size": REPRESENTATION_DIM,
        "classes": NUM_CLASSES,
        "device": str(device),

        "scheduler": {
            "type": TRAINING_DEFAULTS["scheduler"]["type"],
            "T_max": int(epochs),
            "eta_min": float(
                TRAINING_DEFAULTS["scheduler"]["eta_min"]
            ),
        },

        "gradient_clipping": {
            "strategy": clipping_strategy,
            "max_norm": float(max_grad_norm),
        },

        "augmentation": TRAINING_DEFAULTS["augmentation"],

        "feedback_initialization": (
            TRAINING_DEFAULTS["feedback_initialization"]
        ),

        # Correct location for checkpoint selection metadata.
        "checkpoint_selection": (
            TRAINING_DEFAULTS["checkpoint_selection"]
        ),
    }

    return {
        "representation_version": REPRESENTATION_VERSION,
        "dataset": "CIFAR-10",

        "dataset_fingerprint": dataset_fingerprint(
            train_dataset,
            test_dataset,
        ),

        "split_protocol": (
            "official_train_stratified_train_validation_holdout_v1"
        ),

        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "split_indices": indices,

        "sample_counts": {
            "train": len(indices["train"]),
            "validation": len(indices["validation"]),
            "test": len(indices["test"]),
        },

        "class_counts": {
            "train": make_class_counts(
                train_labels[indices["train"]]
            ),
            "validation": make_class_counts(
                train_labels[indices["validation"]]
            ),
            "test": make_class_counts(test_labels),
        },

        "backbone": BACKBONE,
        "backbone_parameter_count": BACKBONE_PARAMETER_COUNT,
        "training": training,
        "preprocessing": PREPROCESSING,

        "representation_layer": (
            "relu(fc1(flatten(avgpool4x4(relu(conv2(relu(conv1(image))))))))"
        ),

        "representation_dim": REPRESENTATION_DIM,
        "representation_dtype": "float32",

        "run_provenance": collect_run_provenance(str(device)),
    }


def _tensor_digest(data: dict[str, Any]) -> str:
    """Hash all saved feature and label tensors."""

    digest = hashlib.sha256()

    for key in (
        "H_train",
        "y_train",
        "H_validation",
        "y_validation",
        "H_test",
        "y_test",
    ):
        if key not in data:
            raise KeyError(
                f"Missing representation tensor: {key}"
            )

        tensor = data[key]

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{key} must be a torch.Tensor."
            )

        tensor = tensor.detach().cpu().contiguous()

        digest.update(key.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))

        digest.update(
            np.asarray(
                tensor.shape,
                dtype=np.int64,
            ).tobytes()
        )

        digest.update(tensor.numpy().tobytes())

    return digest.hexdigest()


def _compatibility_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Exclude nonessential runtime provenance from cache comparison."""

    compatible = dict(metadata)

    compatible.pop("run_provenance", None)

    training = compatible.get("training")

    if isinstance(training, dict):
        training = dict(training)
        training.pop("device", None)
        compatible["training"] = training

    return compatible


def _metadata_mismatch_fields(
    actual: Any,
    expected: Any,
    prefix: str = "",
) -> list[str]:
    """Return metadata paths whose values differ."""

    if isinstance(actual, dict) and isinstance(expected, dict):
        fields = []

        for key in sorted(set(actual) | set(expected)):
            child = f"{prefix}.{key}" if prefix else key

            if key not in actual or key not in expected:
                fields.append(child)
            else:
                fields.extend(
                    _metadata_mismatch_fields(
                        actual[key],
                        expected[key],
                        child,
                    )
                )

        return fields

    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return [prefix]

        fields = []

        for index, (left, right) in enumerate(
            zip(actual, expected)
        ):
            fields.extend(
                _metadata_mismatch_fields(
                    left,
                    right,
                    f"{prefix}[{index}]",
                )
            )

        return fields

    return [] if actual == expected else [prefix]


def representation_path(
    output_dir: Path | str,
    seed: int,
    epochs: int,
    validation_fraction: float,
) -> Path:
    """Build a version-specific representation artifact path."""

    fraction = (
        f"{float(validation_fraction):.4f}".replace(".", "p")
    )

    return (
        Path(output_dir)
        / (
            f"cifar10_bnn_dfa_v4_seed_{seed}_"
            f"epochs_{epochs}_val_{fraction}.pt"
        )
    )


def save_representation(
    path: Path | str,
    data: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Save representation with integrity hashes and overwrite protection."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raise FileExistsError(
            "Refusing to overwrite CIFAR-10 representation artifact: "
            f"{path}"
        )

    saved = dict(data)

    saved["metadata"] = metadata
    saved["representation_sha256"] = _tensor_digest(saved)

    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    saved["artifact_id"] = hashlib.sha256(
        metadata_bytes
        + saved["representation_sha256"].encode("ascii")
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
    expected_metadata: dict[str, Any],
    train_dataset: Any,
    test_dataset: Any,
) -> dict[str, Any]:
    """Load and strictly validate a saved representation artifact."""

    path = Path(path)

    try:
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except Exception as exc:
        raise ValueError(
            "CIFAR-10 representation artifact could not be "
            f"deserialized: {path}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "CIFAR-10 representation artifact is malformed: "
            "expected a dictionary."
        )

    actual_metadata = data.get("metadata")

    if not isinstance(actual_metadata, dict):
        raise ValueError(
            "CIFAR-10 representation artifact is missing metadata."
        )

    mismatch_fields = _metadata_mismatch_fields(
        _compatibility_metadata(actual_metadata),
        _compatibility_metadata(expected_metadata),
    )

    if mismatch_fields:
        details = ", ".join(mismatch_fields[:12])

        if len(mismatch_fields) > 12:
            details += f", and {len(mismatch_fields) - 12} more"

        raise ValueError(
            "CIFAR-10 representation metadata mismatch in: "
            f"{details}"
        )

    try:
        actual_tensor_digest = _tensor_digest(data)

    except (
        KeyError,
        AttributeError,
        TypeError,
        RuntimeError,
    ) as exc:
        raise ValueError(
            "CIFAR-10 representation artifact is missing or "
            "contains invalid tensors."
        ) from exc

    if data.get("representation_sha256") != actual_tensor_digest:
        raise ValueError(
            "CIFAR-10 representation artifact tensor checksum is invalid."
        )

    metadata_bytes = json.dumps(
        actual_metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    expected_id = hashlib.sha256(
        metadata_bytes
        + data["representation_sha256"].encode("ascii")
    ).hexdigest()

    if data.get("artifact_id") != expected_id:
        raise ValueError(
            "CIFAR-10 representation artifact identifier is invalid."
        )

    # Validate checkpoint-selection information.
    summary = data.get("training_summary")

    if not isinstance(summary, dict):
        raise ValueError(
            "v3 artifact is missing training_summary."
        )

    selected_epoch = summary.get("selected_epoch")

    if (
        not isinstance(selected_epoch, int)
        or isinstance(selected_epoch, bool)
        or not 1 <= selected_epoch <= expected_metadata["training"]["epochs"]
    ):
        raise ValueError(
            "training_summary contains an invalid selected_epoch."
        )

    selected_loss = summary.get("selected_validation_loss")
    selected_accuracy = summary.get("selected_validation_accuracy")

    if (
        not isinstance(selected_loss, (int, float))
        or not np.isfinite(selected_loss)
        or selected_loss < 0
    ):
        raise ValueError(
            "training_summary contains an invalid validation loss."
        )

    if (
        not isinstance(selected_accuracy, (int, float))
        or not np.isfinite(selected_accuracy)
        or not 0 <= selected_accuracy <= 100
    ):
        raise ValueError(
            "training_summary contains an invalid validation accuracy."
        )

    indices = expected_metadata["split_indices"]

    train_y = torch.as_tensor(
        _numpy(train_dataset.targets)[indices["train"]],
        dtype=torch.int64,
    )

    validation_y = torch.as_tensor(
        _numpy(train_dataset.targets)[indices["validation"]],
        dtype=torch.int64,
    )

    test_y = torch.as_tensor(
        _numpy(test_dataset.targets)[indices["test"]],
        dtype=torch.int64,
    )

    expected_labels_by_split = {
        "train": train_y,
        "validation": validation_y,
        "test": test_y,
    }

    for name, expected_labels in expected_labels_by_split.items():
        feature_key = f"H_{name}"
        label_key = f"y_{name}"

        features = data.get(feature_key)
        labels = data.get(label_key)

        if not isinstance(features, torch.Tensor):
            raise ValueError(
                f"{feature_key} is not a tensor."
            )

        if not isinstance(labels, torch.Tensor):
            raise ValueError(
                f"{label_key} is not a tensor."
            )

        expected_shape = (
            len(expected_labels),
            REPRESENTATION_DIM,
        )

        if (
            features.ndim != 2
            or tuple(features.shape) != expected_shape
        ):
            raise ValueError(
                f"{feature_key} has incompatible shape: "
                f"{tuple(features.shape)}; expected {expected_shape}."
            )

        if features.dtype != torch.float32:
            raise ValueError(
                f"{feature_key} must have dtype torch.float32."
            )

        if labels.dtype != torch.int64:
            raise ValueError(
                f"{label_key} must have dtype torch.int64."
            )

        if labels.shape != expected_labels.shape:
            raise ValueError(
                f"{label_key} has an incompatible shape."
            )

        if not torch.equal(labels, expected_labels):
            raise ValueError(
                f"{label_key} is misaligned with saved dataset indices."
            )

        if not torch.isfinite(features).all().item():
            raise ValueError(
                f"{feature_key} contains non-finite values."
            )

    return data