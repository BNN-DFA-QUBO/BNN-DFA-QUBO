"""CIFAR-10 data loading, preprocessing, and stratified splitting utilities.

This module manages the standard CIFAR-10 dataset (50,000 train, 10,000 test)
and provides:
1. Stratified 45,000 train / 5,000 validation split.
2. Data augmentation (RandomCrop, RandomHorizontalFlip) strictly for backbone training.
3. Deterministic transforms (ToTensor + Normalize) for validation, test, and feature extraction.
4. Support for CPU, CUDA, and Apple Silicon MPS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_DATA_DIR = Path("data/cifar10/raw")


def get_device() -> torch.device:
    """Return highest-priority available compute device (MPS > CUDA > CPU)."""
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_train_transform() -> transforms.Compose:
    """Augmented transforms for BNN+DFA backbone training only."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    """Deterministic transforms for validation, testing, and feature extraction."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


class TransformedSubset(Dataset):
    """Wraps a Dataset subset with a specific transform, ensuring label alignment."""

    def __init__(self, dataset: Dataset, indices: list[int], transform: transforms.Compose):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        # dataset[real_idx] returns (image, target) where image is PIL Image if transform is None
        item = self.dataset[real_idx]
        image, target = item[0], item[1]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target)


def get_cifar10_raw_datasets(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    download: bool = True,
) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Load base CIFAR-10 torchvision datasets without default transforms."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_raw = datasets.CIFAR10(root=str(data_path), train=True, download=download, transform=None)
    test_raw = datasets.CIFAR10(root=str(data_path), train=False, download=download, transform=None)
    return train_raw, test_raw


def get_cifar10_split_indices(
    train_dataset: datasets.CIFAR10,
    test_dataset: datasets.CIFAR10,
    seed: int,
    validation_fraction: float = 0.1,
) -> Dict[str, list[int]]:
    """Compute deterministic stratified split indices."""
    targets = np.asarray(train_dataset.targets, dtype=np.int64).reshape(-1)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_dataset)),
        test_size=float(validation_fraction),
        random_state=int(seed),
        stratify=targets,
    )
    return {
        "train": train_idx.astype(np.int64).tolist(),
        "validation": val_idx.astype(np.int64).tolist(),
        "test": np.arange(len(test_dataset), dtype=np.int64).tolist(),
    }


def get_cifar10_datasets(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    seed: int = 42,
    validation_fraction: float = 0.1,
    download: bool = True,
) -> Dict[str, Any]:
    """Return dictionary of configured dataset views for CIFAR-10.

    Views:
        - train_augmented: for backbone training (augmented)
        - train_deterministic: for feature extraction (unaugmented)
        - validation: for validation and tuning (deterministic)
        - test: for final reporting only (deterministic)
    """
    train_raw, test_raw = get_cifar10_raw_datasets(data_dir, download=download)
    indices = get_cifar10_split_indices(train_raw, test_raw, seed, validation_fraction)

    train_aug_transform = get_train_transform()
    eval_transform = get_eval_transform()

    train_augmented = TransformedSubset(train_raw, indices["train"], train_aug_transform)
    train_deterministic = TransformedSubset(train_raw, indices["train"], eval_transform)
    val_dataset = TransformedSubset(train_raw, indices["validation"], eval_transform)
    test_dataset = TransformedSubset(test_raw, indices["test"], eval_transform)

    return {
        "train_raw": train_raw,
        "test_raw": test_raw,
        "split_indices": indices,
        "train_augmented": train_augmented,
        "train_deterministic": train_deterministic,
        "validation": val_dataset,
        "test": test_dataset,
    }


def get_cifar10_loaders(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    seed: int = 42,
    validation_fraction: float = 0.1,
    batch_size: int = 64,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    download: bool = True,
) -> Dict[str, Any]:
    """Create PyTorch DataLoaders for all four required data splits/views."""
    cifar_datasets = get_cifar10_datasets(
        data_dir=data_dir,
        seed=seed,
        validation_fraction=validation_fraction,
        download=download,
    )

    train_loader = DataLoader(
        cifar_datasets["train_augmented"],
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
        num_workers=num_workers,
    )
    train_eval_loader = DataLoader(
        cifar_datasets["train_deterministic"],
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        cifar_datasets["validation"],
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        cifar_datasets["test"],
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return {
        "datasets": cifar_datasets,
        "train_loader": train_loader,
        "train_eval_loader": train_eval_loader,
        "validation_loader": val_loader,
        "test_loader": test_loader,
        "split_indices": cifar_datasets["split_indices"],
    }
