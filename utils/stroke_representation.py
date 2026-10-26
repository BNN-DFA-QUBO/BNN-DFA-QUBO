import torch
from pathlib import Path
import hashlib
import json
import warnings

from utils.stroke_config import PREPROCESSING_VERSION
from utils.stroke_data import METADATA_PATH, RAW_PATH
from utils.run_provenance import collect_run_provenance

LEGACY_REPRESENTATION_VERSION = "stroke-representation-v2"
REPRESENTATION_VERSION = "stroke-representation-v3"


def _representation_digest(data):
    """Hash canonical tensor bytes plus shape/dtype and row/protocol metadata."""
    digest = hashlib.sha256()
    for key in ("H_train", "y_train", "H_val", "y_val", "H_test", "y_test"):
        tensor = torch.as_tensor(data[key]).detach().cpu().contiguous()
        digest.update(key.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    bound_metadata = {
        "data_metadata": data.get("data_metadata", {}),
        "feature_names": data.get("feature_names", []),
        "training_config": data.get("training_config", {}),
        "training_seed": data.get("training_seed", data.get("seed")),
        "input_size": data.get("input_size"),
        "hidden_size": data.get("hidden_size"),
        "run_provenance": data.get("run_provenance", {}),
    }
    digest.update(json.dumps(bound_metadata, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    return digest.hexdigest()


def _validate_representation_arrays(data):
    metadata = data.get("data_metadata", {})
    expected_hidden = int(data.get("hidden_size", -1))
    for split, id_key in (("train", "resampled_train_row_ids"),
                          ("val", "validation"), ("test", "test")):
        hidden = torch.as_tensor(data[f"H_{split}"])
        labels = torch.as_tensor(data[f"y_{split}"]).reshape(-1)
        if hidden.ndim != 2 or hidden.shape[0] == 0 or hidden.shape[1] == 0:
            raise ValueError(f"Representation H_{split} must be a non-empty 2D matrix")
        if hidden.shape[1] != expected_hidden or hidden.shape[0] != len(labels):
            raise ValueError(f"Representation H_{split}/y_{split} dimensions do not match saved configuration")
        if not torch.isfinite(hidden).all():
            raise ValueError(f"Representation H_{split} contains non-finite values")
        if not torch.isfinite(labels).all() or not torch.isin(labels, torch.tensor([0.0, 1.0])).all():
            raise ValueError(f"Representation y_{split} must contain finite binary labels")
        ids = (metadata.get("resampled_train_row_ids", []) if split == "train"
               else metadata.get("split_row_ids", {}).get(id_key, []))
        if ids and len(ids) != len(labels):
            raise ValueError(f"Representation y_{split} does not align with saved row identities")

        count_key = {"train": "resampled_train_class_counts", "val": "validation_class_counts",
                     "test": "test_class_counts"}[split]
        labels_long = labels.long()
        counts = {str(label): int((labels_long == label).sum()) for label in (0, 1)}
        if metadata.get(count_key) != counts:
            raise ValueError(f"Representation y_{split} class counts disagree with preprocessing metadata")

    split_ids = metadata.get("split_row_ids", {})
    train_ids = split_ids.get("train", [])
    val_ids = split_ids.get("validation", [])
    test_ids = split_ids.get("test", [])
    resampled_ids = metadata.get("resampled_train_row_ids", [])
    if not all((train_ids, val_ids, test_ids, resampled_ids)):
        raise ValueError("Representation metadata is missing split row identities")
    if (set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids)
            or set(val_ids) & set(test_ids)):
        raise ValueError("Representation split row identities overlap")
    if not set(resampled_ids) <= set(train_ids):
        raise ValueError("Resampled training rows include identities outside the original training split")

    row_labels = metadata.get("split_row_labels")
    if row_labels is None:
        return False
    for split, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids)):
        labels = row_labels.get(split)
        if labels is None or len(labels) != len(ids):
            raise ValueError(f"Representation split labels do not align with {split} row identities")
    train_label_map = dict(zip(train_ids, row_labels["train"]))
    expected_labels = {
        "train": [train_label_map[row_id] for row_id in resampled_ids],
        "val": row_labels["validation"],
        "test": row_labels["test"],
    }
    for split, labels in expected_labels.items():
        actual = torch.as_tensor(data[f"y_{split}"]).reshape(-1).long()
        expected = torch.as_tensor(labels).reshape(-1).long()
        if not torch.equal(actual, expected):
            raise ValueError(f"Representation y_{split} labels do not correspond to stored row identities")
    return True


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
    data["run_provenance"] = result.get("run_provenance") or collect_run_provenance(str(result.get("device", "unavailable")))

    if "dfa_feedback" in result:
        data["dfa_feedback"] = result["dfa_feedback"]
    if len(names) != int(data["input_size"]):
        raise ValueError("Representation feature names do not match input dimension")
    if int(data["H_train"].shape[1]) != int(data["hidden_size"]):
        raise ValueError("Representation hidden dimension does not match saved configuration")
    if not _validate_representation_arrays(data):
        raise ValueError("Cannot save a verified Stroke representation without persisted labels for each split row ID")
    data["representation_sha256"] = _representation_digest(data)
    data["integrity_status"] = "content_and_row_id_verified"
    created = False
    try:
        with path.open("xb") as output:
            created = True
            torch.save(data, output)
    except Exception:
        if created and path.exists():
            path.unlink()
        raise
    return data


def load_representation(path, expected_metadata=None, expected_seed=None,
                        expected_training_config=None, expected_hidden_size=None):
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )
    version = data.get("representation_version")
    if version not in {REPRESENTATION_VERSION, LEGACY_REPRESENTATION_VERSION}:
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
    elif expected_metadata is None:
        warnings.warn(
            "No current Stroke preprocessing metadata is available; dataset and split compatibility are verified only from the artifact itself",
            RuntimeWarning,
        )
    if expected_metadata is not None:
        missing_expected_metadata = []
        for key in ("dataset_sha256", "split_seed", "preprocessing_version", "preprocessing_config_hash"):
            if key not in metadata or key not in expected_metadata:
                missing_expected_metadata.append(key)
            elif metadata[key] != expected_metadata[key]:
                raise ValueError(f"Representation {key} does not match the requested experiment")
        if missing_expected_metadata:
            warnings.warn(
                "Strict Stroke representation compatibility could not be established because metadata is missing: "
                + ", ".join(missing_expected_metadata), RuntimeWarning,
            )
    _validate_representation_arrays(data)
    names = data.get("feature_names", [])
    if names and len(names) != data.get("input_size"):
        raise ValueError("Representation feature names do not match its saved input dimension")
    if names and metadata.get("feature_names") != names:
        raise ValueError("Representation schema differs from preprocessing metadata")
    if metadata.get("input_size") != data.get("input_size"):
        raise ValueError("Representation input size differs from preprocessing metadata")
    if data.get("hidden_size") != data["H_train"].shape[1]:
        raise ValueError("Representation hidden size does not match its saved arrays")
    if expected_seed is not None:
        stored_seed = data.get("training_seed", data.get("seed"))
        if stored_seed is None:
            warnings.warn(
                "Strict Stroke representation seed compatibility could not be established: training seed metadata is missing",
                RuntimeWarning,
            )
        elif int(stored_seed) != int(expected_seed):
            raise ValueError("Representation training seed does not match the requested seed")
    if expected_hidden_size is not None and int(data["hidden_size"]) != int(expected_hidden_size):
        raise ValueError("Representation hidden dimension does not match the requested model configuration")
    if expected_training_config is not None:
        actual_config = data.get("training_config", {})
        missing_config = []
        for key, value in expected_training_config.items():
            if key not in actual_config:
                missing_config.append(key)
            elif actual_config[key] != value:
                raise ValueError(f"Representation training configuration field {key!r} does not match")
        if missing_config:
            warnings.warn(
                "Strict Stroke backbone configuration compatibility could not be established because metadata is missing: "
                + ", ".join(missing_config), RuntimeWarning,
            )

    if version == LEGACY_REPRESENTATION_VERSION:
        warnings.warn(
            "Loading legacy Stroke representation without a content hash; feature content and exact row-label correspondence are unverified. Regenerate the representation for full integrity checks.",
            RuntimeWarning,
        )
        data["integrity_status"] = "legacy_unverified"
    else:
        if "split_row_labels" not in metadata:
            raise ValueError("Hashed Stroke representation is missing split labels needed to bind labels to row identities")
        if data.get("integrity_status") != "content_and_row_id_verified":
            raise ValueError("Stroke representation integrity status is missing or unsupported")
        actual_hash = _representation_digest(data)
        if data.get("representation_sha256") != actual_hash:
            raise ValueError("Stroke representation content hash mismatch; feature, label, or provenance data changed")
    return data


def load_stroke_head_representation(path, expected_seed=None, expected_metadata=None):
    """Load for a project head using its known hidden dimension, not head settings.

    The optional seed is the representation/backbone seed. Head initialization
    seeds must not be passed here. A caller with a requested backbone config
    should use ``load_representation`` and pass that config explicitly; heads
    that only consume a saved representation do not assume an epoch count.
    """
    from utils.stroke_config import HIDDEN_SIZE

    return load_representation(
        path,
        expected_metadata=expected_metadata,
        expected_seed=expected_seed,
        expected_hidden_size=HIDDEN_SIZE,
    )
