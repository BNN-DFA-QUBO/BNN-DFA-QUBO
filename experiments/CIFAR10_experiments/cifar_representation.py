"""Create or safely reuse one fixed CIFAR-10 BNN+DFA representation per seed/config."""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from experiments.CIFAR10_experiments.cifar_bnn_dfa import (
    collect_hidden_features,
    train_dfa_epoch,
)
from models.cifar_bnn import CIFAR_BNN
from models.cifar_dfa import CIFARDFA
from utils.CIFAR10.data import (
    DEFAULT_DATA_DIR,
    TransformedSubset,
    get_cifar10_raw_datasets,
    get_eval_transform,
    get_train_transform,
)
from utils.CIFAR10.representation import (
    DEFAULT_REPRESENTATION_DIR,
    build_metadata,
    load_representation,
    representation_path,
    save_representation,
)
from utils.MNIST.seed import set_seed


def load_or_train_cifar10_representation(
    seed: int,
    epochs: int = 20,
    validation_fraction: float = 0.1,
    batch_size: int = 64,
    eval_batch_size: int = 256,
    learning_rate: float = 0.001,
    device: str | torch.device = "cpu",
    output_dir: Path | str = DEFAULT_REPRESENTATION_DIR,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    train_dataset: Any = None,
    test_dataset: Any = None,
) -> Tuple[Dict[str, Any], Path, str]:
    """Load an existing validated CIFAR-10 representation, or train one and cache it."""
    dev = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if train_dataset is None or test_dataset is None:
        train_raw, test_raw = get_cifar10_raw_datasets(data_dir=data_dir, download=True)
    else:
        train_raw, test_raw = train_dataset, test_dataset

    metadata = build_metadata(
        train_dataset=train_raw,
        test_dataset=test_raw,
        seed=seed,
        epochs=epochs,
        validation_fraction=validation_fraction,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=str(dev),
    )
    path = representation_path(output_dir, seed, epochs, validation_fraction)

    def load_cached():
        try:
            return load_representation(path, metadata, train_raw, test_raw)
        except Exception as exc:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            recovery_dir = output_dir / f"rebuild_seed_{seed}_epochs_{epochs}_{stamp}"
            command = (
                f'"{sys.executable}" -m experiments.CIFAR10_experiments.run_cifar10_shared_validation '
                f"--seed {seed} --epochs {epochs} --validation-fraction {validation_fraction:g} "
                f'--representation-dir "{recovery_dir}"'
            )
            raise RuntimeError(
                f"CIFAR-10 representation cache at {path} could not be reused: {exc}. "
                f"The existing artifact was left unchanged. To build a fresh cache, run: {command}"
            ) from exc

    if path.exists():
        data = load_cached()
        return data, path, "reused"

    # Setup loaders
    indices = metadata["split_indices"]
    train_aug_transform = get_train_transform()
    eval_transform = get_eval_transform()

    train_aug_subset = TransformedSubset(train_raw, indices["train"], train_aug_transform)
    train_det_subset = TransformedSubset(train_raw, indices["train"], eval_transform)
    val_subset = TransformedSubset(train_raw, indices["validation"], eval_transform)
    test_subset = TransformedSubset(test_raw, indices["test"], eval_transform)

    train_loader = DataLoader(
        train_aug_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
        num_workers=0,
    )
    train_eval_loader = DataLoader(
        train_det_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        val_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    set_seed(seed)
    model = CIFAR_BNN().to(dev)
    dfa = CIFARDFA(device=dev, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    training_started = time.perf_counter()

    for epoch in range(epochs):
        loss, acc = train_dfa_epoch(model, train_loader, optimizer, dfa, dev)
        print(f"CIFAR Backbone Epoch {epoch+1:02d}/{epochs:02d}: loss={loss:.4f}, acc={acc:.2f}%")

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    training_seconds = time.perf_counter() - training_started

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    extraction_started = time.perf_counter()

    representations = {}
    for split_name, loader in (
        ("train", train_eval_loader),
        ("validation", validation_loader),
        ("test", test_loader),
    ):
        feats, labels = collect_hidden_features(model, loader, dev)
        representations[f"H_{split_name}"] = feats.detach().cpu().float().contiguous()
        representations[f"y_{split_name}"] = labels.detach().cpu().long().contiguous()

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    extraction_seconds = time.perf_counter() - extraction_started

    representations["training_seconds"] = training_seconds
    representations["feature_extraction_seconds"] = extraction_seconds

    try:
        save_representation(path, representations, metadata)
    except FileExistsError:
        data = load_cached()
        return data, path, "reused"

    data = load_representation(path, metadata, train_raw, test_raw)
    return data, path, "trained"
