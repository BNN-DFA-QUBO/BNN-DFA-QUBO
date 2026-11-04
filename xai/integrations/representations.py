"""Load validated frozen representations for XAI."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from xai.core.contracts import RepresentationBundle


def _np(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_dataset_representation(
    dataset: str,
    *,
    seed: int,
    epochs: int | None,
    validation_fraction: float = 0.15,
    device: str | None = None,
    artifact_dir: Path | str = Path("data/xai_artifacts"),
    overwrite: bool = False,
) -> RepresentationBundle:

    dataset = dataset.lower()

    supported = {
        "mnist",
        "bank",
        "stroke",
        "cifar10",
        "resnet18_cifar10",
    }

    if dataset not in supported:
        raise ValueError(f"Unsupported dataset: {dataset}")

    dev = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    root = Path(artifact_dir) / dataset / "representations"
    root.mkdir(parents=True, exist_ok=True)

    if dataset == "mnist":

        from utils.MNIST.data import get_mnist_loaders
        from experiments.MNIST_experiments.mnist_representation import (
            load_or_train_mnist_representation,
        )

        train_loader, test_loader = get_mnist_loaders(batch_size=64)

        payload, path, source = load_or_train_mnist_representation(
            train_loader.dataset,
            test_loader.dataset,
            seed,
            epochs or 16,
            validation_fraction,
            dev,
            root,
        )

        x_train = payload["H_train"]
        y_train = payload["y_train"]
        x_val = payload["H_validation"]
        y_val = payload["y_validation"]
        x_test = payload["H_test"]
        y_test = payload["y_test"]

        names = [
            f"latent_{i:03d}"
            for i in range(int(x_train.shape[1]))
        ]

        meta = payload.get("metadata", {})

    elif dataset == "bank":

        from utils.bank.data import get_bank_datasets
        from experiments.bank_experiments.bank_representation import (
            load_or_train_bank_representation,
        )

        train_ds, val_ds, test_ds = get_bank_datasets()

        payload, path, source, _, _ = load_or_train_bank_representation(
            train_ds,
            val_ds,
            test_ds,
            seed=seed,
            epochs=epochs or 15,
            device=dev,
            output_dir=root,
        )

        x_train = payload["H_train"]
        y_train = payload["y_train"]
        x_val = payload["H_val"]
        y_val = payload["y_val"]
        x_test = payload["H_test"]
        y_test = payload["y_test"]

        names = [
            f"latent_{i:03d}"
            for i in range(int(x_train.shape[1]))
        ]

        meta = {}

    elif dataset == "stroke":

        from utils.stroke_data import DATA_DIR as STROKE_DATA_DIR
        from utils.stroke_data import prepare_stroke_data
        from utils.stroke_config import SPLIT_SEED, BALANCING_METHOD

        from experiments.stroke_experiments.run_stroke_shared_validation import (
            load_or_train_stroke_representation,
        )

        prepare_stroke_data(
            split_seed=SPLIT_SEED,
            balance_method=BALANCING_METHOD,
            output_dir=root,
            overwrite=overwrite,
        )

        payload, path, source = load_or_train_stroke_representation(
            seed=seed,
            split_seed=SPLIT_SEED,
            balancing_method=BALANCING_METHOD,
            output_dir=root,
            overwrite=overwrite,
        )

        x_train = payload["H_train"]
        y_train = payload["y_train"]
        x_val = payload["H_val"]
        y_val = payload["y_val"]
        x_test = payload["H_test"]
        y_test = payload["y_test"]

        names = [
            f"latent_{i:03d}"
            for i in range(int(x_train.shape[1]))
        ]

        meta = {
            k: payload.get(k)
            for k in (
                "seed",
                "training_seed",
                "split_seed",
                "integrity_status",
                "representation_sha256",
            )
            if k in payload
        }

        meta["raw_data_directory"] = str(STROKE_DATA_DIR)

    elif dataset == "cifar10":

        # Existing BNN-DFA CIFAR-10 path remains unchanged.
        from experiments.CIFAR10_experiments.cifar_representation import (
            load_or_train_cifar10_representation,
        )

        payload, path, source = load_or_train_cifar10_representation(
            seed=seed,
            epochs=epochs or 20,
            validation_fraction=validation_fraction,
            device=dev,
            output_dir=root,
        )

        x_train = payload["H_train"]
        y_train = payload["y_train"]
        x_val = payload["H_validation"]
        y_val = payload["y_validation"]
        x_test = payload["H_test"]
        y_test = payload["y_test"]

        names = [
            f"latent_{i:03d}"
            for i in range(int(x_train.shape[1]))
        ]

        meta = payload.get("metadata", {})

    else:

        # Dedicated ResNet18 CIFAR-10 representation.
        from xai.integrations.resnet18_representation import (
            load_or_create_resnet18_representation,
        )

        payload, path, source = load_or_create_resnet18_representation(
            seed=seed,
            device=dev,
            artifact_dir=artifact_dir,
            validation_fraction=0.1,
            overwrite=overwrite,
        )

        x_train = payload["H_train"]
        y_train = payload["y_train"]

        x_val = payload["H_validation"]
        y_val = payload["y_validation"]

        x_test = payload["H_test"]
        y_test = payload["y_test"]

        if x_train.shape[1] != 512:
            raise ValueError(
                f"Expected 512 ResNet18 features, got {x_train.shape[1]}"
            )

        names = [
            f"resnet18_feature_{i:03d}"
            for i in range(512)
        ]

        meta = payload.get("metadata", {})

    path = Path(path)

    return RepresentationBundle(
        dataset=dataset,

        X_train=_np(x_train).astype(np.float64),
        y_train=_np(y_train).astype(int).reshape(-1),

        X_validation=_np(x_val).astype(np.float64),
        y_validation=_np(y_val).astype(int).reshape(-1),

        X_test=_np(x_test).astype(np.float64),
        y_test=_np(y_test).astype(int).reshape(-1),

        feature_names=names,

        representation_path=str(path.resolve()),
        representation_sha256=_hash(path),
        representation_source=str(source),

        metadata=meta if isinstance(meta, dict) else {},
    )