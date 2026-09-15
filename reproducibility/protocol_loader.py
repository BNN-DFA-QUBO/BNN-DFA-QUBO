from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


class ProtocolSampler(Sampler):
    """Sampler that replays the exact index order stored in the protocol."""

    def __init__(self, protocol_dir, num_samples, mode="train"):
        self.protocol_dir = Path(protocol_dir)
        self.num_samples = int(num_samples)
        self.mode = mode
        self.iteration = 0

        if self.mode not in {"train", "test"}:
            raise ValueError("mode must be 'train' or 'test'")

        if self.mode == "test":
            self.indices = self._load_test_indices()

    def _load_test_indices(self):
        path = self.protocol_dir / "test_indices.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing test protocol: {path}")

        indices = np.load(path)
        self._validate(indices, path)
        return indices

    def _load_train_indices(self, epoch):
        path = self.protocol_dir / "train" / f"epoch_{epoch:03d}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing training protocol: {path}")

        indices = np.load(path)
        self._validate(indices, path)
        return indices

    def _validate(self, indices, path):
        if len(indices) != self.num_samples:
            raise ValueError(
                f"{path} contains {len(indices)} indices; "
                f"expected {self.num_samples}"
            )

        expected = np.arange(self.num_samples)
        if np.any(indices < 0) or np.any(indices >= self.num_samples):
            raise ValueError(f"{path} contains out-of-range dataset indices")

        if not np.array_equal(np.sort(indices), expected):
            raise ValueError(f"{path} is not a valid permutation")

    def __iter__(self):
        if self.mode == "test":
            indices = self.indices
        else:
            # The experiments train for the protocol's number of epochs.
            # Any later pass over train_loader (e.g. hidden-feature collection)
            # uses epoch 0 as a canonical fixed order.
            epoch = self.iteration
            try:
                indices = self._load_train_indices(epoch)
            except FileNotFoundError:
                indices = self._load_train_indices(0)

            self.iteration += 1

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


def build_protocol_loaders(
    original_get_mnist_loaders,
    protocol_dir,
    batch_size=64,
):
    """Build protocol-controlled loaders while preserving the original datasets/transforms."""
    original_train_loader, original_test_loader = original_get_mnist_loaders(
        batch_size=batch_size
    )

    train_dataset = original_train_loader.dataset
    test_dataset = original_test_loader.dataset

    train_sampler = ProtocolSampler(
        protocol_dir=protocol_dir,
        num_samples=len(train_dataset),
        mode="train",
    )
    test_sampler = ProtocolSampler(
        protocol_dir=protocol_dir,
        num_samples=len(test_dataset),
        mode="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


def load_manifest(protocol_dir):
    path = Path(protocol_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing protocol manifest: {path}")

    with open(path, "r") as f:
        return json.load(f)
