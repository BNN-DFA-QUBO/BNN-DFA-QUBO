"""Frozen 512-D ResNet18 CIFAR-10 representation for XAI."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEFAULT_CHECKPOINT_DIR = Path("data/resnet18_cifar10")
DEFAULT_DATA_DIR = Path("data/cifar10/raw")


class IndexedDataset(Dataset):
    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.base[int(self.indices[index])]
        return self.transform(image), label


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _load_checkpoint(path: Path):
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid ResNet18 checkpoint format.")

    if "state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain the expected 'state_dict' key."
        )

    return checkpoint


def _extract_features(model, loader, device):
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            output = model(images)

            if output.ndim != 2 or output.shape[1] != 512:
                raise RuntimeError(
                    f"Expected ResNet18 512-D features, got {tuple(output.shape)}"
                )

            features.append(output.detach().cpu().float())
            labels.append(targets.cpu().long())

    return torch.cat(features), torch.cat(labels)


def load_or_create_resnet18_representation(
    seed: int,
    device: str | None = None,
    artifact_dir: Path | str = Path("data/xai_artifacts"),
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    validation_fraction: float = 0.1,
    overwrite: bool = False,
):
    """
    Load the existing ResNet18 checkpoint and extract standardized 512-D
    penultimate-layer features.

    The split and preprocessing reproduce run_resnet18_qubo.py.
    """

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    # The checkpoint does not store validation_fraction.
    # The original benchmark uses 0.1, so do not silently change it.
    if not np.isclose(validation_fraction, 0.1):
        raise ValueError(
            "The existing ResNet18 checkpoint was trained with "
            "validation_fraction=0.1. Use the same value."
        )

    checkpoint_path = (
        Path(checkpoint_dir)
        / f"resnet18_cifar10_seed_{seed}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"ResNet18 checkpoint not found: {checkpoint_path}\n"
            f"Train it first using:\n"
            f"python -m experiments.CIFAR10_experiments.run_resnet18_qubo "
            f"--seed {seed} --epochs 20"
        )

    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)

    if "seed" in checkpoint and int(checkpoint["seed"]) != int(seed):
        raise ValueError(
            f"Checkpoint seed mismatch: expected {seed}, "
            f"found {checkpoint['seed']}."
        )

    if device is None:
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    root = (
        Path(artifact_dir)
        / "resnet18_cifar10"
        / "representations"
    )
    root.mkdir(parents=True, exist_ok=True)

    artifact_path = root / f"resnet18_seed_{seed}.pt"

    expected_metadata = {
        "dataset": "resnet18_cifar10",
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "validation_fraction": 0.1,
        "feature_dimension": 512,
        "split_protocol": "train_test_split_random_state_seed_stratified",
        "preprocessing": "Resize96_ToTensor_ImageNetNormalize",
    }

    # Reuse only a cache tied to the exact checkpoint and protocol.
    if artifact_path.exists() and not overwrite:
        try:
            cached = torch.load(
                artifact_path,
                map_location="cpu",
                weights_only=False,
            )

            if cached.get("metadata") == expected_metadata:
                required = (
                    "H_train", "y_train",
                    "H_validation", "y_validation",
                    "H_test", "y_test",
                )

                if all(key in cached for key in required):
                    return cached, artifact_path, "reused"

        except Exception:
            pass

    # Same evaluation transform as run_resnet18_qubo.py.
    eval_transform = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    raw_train = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
    )

    raw_test = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
    )

    y_all = np.asarray(raw_train.targets)

    # Exactly reproduce the original benchmark split.
    train_indices, val_indices = train_test_split(
        np.arange(len(raw_train)),
        test_size=0.1,
        random_state=int(seed),
        stratify=y_all,
    )

    test_indices = np.arange(len(raw_test))

    train_dataset = IndexedDataset(
        raw_train, train_indices, eval_transform
    )

    val_dataset = IndexedDataset(
        raw_train, val_indices, eval_transform
    )

    test_dataset = IndexedDataset(
        raw_test, test_indices, eval_transform
    )

    batch_size = 256

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Load the exact fine-tuned model architecture.
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    # Extract 512-dimensional penultimate features.
    model.fc = torch.nn.Identity()

    model = model.to(device)
    model.eval()

    H_train, y_train = _extract_features(
        model, train_loader, device
    )

    H_validation, y_validation = _extract_features(
        model, val_loader, device
    )

    H_test, y_test = _extract_features(
        model, test_loader, device
    )

    # This must match the original ResNet18 benchmark.
    # Statistics are fitted exclusively on training features.
    feature_mean = H_train.mean(
        dim=0,
        keepdim=True,
    )

    feature_std = H_train.std(
        dim=0,
        keepdim=True,
    ).clamp_min(1e-5)

    H_train = (H_train - feature_mean) / feature_std
    H_validation = (H_validation - feature_mean) / feature_std
    H_test = (H_test - feature_mean) / feature_std

    payload = {
        "H_train": H_train.contiguous(),
        "y_train": y_train,
        "H_validation": H_validation.contiguous(),
        "y_validation": y_validation,
        "H_test": H_test.contiguous(),
        "y_test": y_test,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "metadata": expected_metadata,
        "checkpoint_epochs": checkpoint.get("epochs"),
        "best_validation_accuracy": checkpoint.get(
            "best_validation_accuracy"
        ),
    }

    temporary_path = artifact_path.with_suffix(".tmp")

    torch.save(payload, temporary_path)
    os.replace(temporary_path, artifact_path)

    # Verify the saved representation before returning it.
    saved = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=False,
    )

    if saved["H_train"].shape[1] != 512:
        raise RuntimeError("Saved representation is not 512-dimensional.")

    if saved["H_validation"].shape[1] != 512:
        raise RuntimeError("Validation representation dimension mismatch.")

    if saved["H_test"].shape[1] != 512:
        raise RuntimeError("Test representation dimension mismatch.")

    return saved, artifact_path, "created"