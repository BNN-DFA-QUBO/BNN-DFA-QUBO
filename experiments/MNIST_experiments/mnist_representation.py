"""Create or safely reuse one fixed MNIST BNN+DFA representation per seed/config."""
from __future__ import annotations

import time
from pathlib import Path
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

from experiments.MNIST_experiments.bnn_dfa_qubo import collect_hidden_features, train_dfa
from models.bnn import BNN
from models.dfa import DFAClassifier
from utils.MNIST.representation import (
    DEFAULT_REPRESENTATION_DIR,
    build_metadata,
    load_representation,
    representation_path,
    save_representation,
)
from utils.MNIST.seed import set_seed


def load_or_train_mnist_representation(
    train_dataset,
    test_dataset,
    seed,
    epochs=16,
    validation_fraction=0.15,
    device="cpu",
    output_dir=DEFAULT_REPRESENTATION_DIR,
):
    metadata = build_metadata(train_dataset, test_dataset, seed, epochs, validation_fraction,
                              device=device)
    path = representation_path(output_dir, seed, epochs, validation_fraction)

    def load_cached():
        try:
            return load_representation(path, metadata, train_dataset, test_dataset)
        except Exception as exc:
            recovery_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            recovery_fraction = f"{float(validation_fraction):.4f}".replace(".", "p")
            recovery_dir = Path(output_dir) / (
                f"rebuild_seed_{seed}_epochs_{epochs}_val_{recovery_fraction}_{recovery_stamp}"
            )
            command = (
                f'"{sys.executable}" -m experiments.MNIST_experiments.run_mnist_shared_validation '
                f"--seed {seed} --epochs {epochs} --validation-fraction {validation_fraction:g} "
                f"--representation-dir \"{recovery_dir}\""
            )
            raise RuntimeError(
                f"MNIST representation cache at {path} could not be reused: {exc}. "
                f"The existing artifact was left unchanged. To build a fresh, isolated cache, run: {command}"
            ) from exc

    if path.exists():
        data = load_cached()
        return data, path, "reused"

    indices = metadata["split_indices"]
    train_loader = DataLoader(
        Subset(train_dataset, indices["train"]), batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)), num_workers=0,
    )
    train_eval_loader = DataLoader(Subset(train_dataset, indices["train"]), batch_size=256,
                                   shuffle=False, num_workers=0)
    validation_loader = DataLoader(Subset(train_dataset, indices["validation"]), batch_size=256,
                                   shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(test_dataset, indices["test"]), batch_size=256,
                             shuffle=False, num_workers=0)

    set_seed(seed)
    model = BNN().to(device)
    dfa = DFAClassifier(hidden_size=128, output_size=10, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    started = time.perf_counter()
    for epoch in range(epochs):
        loss, accuracy = train_dfa(model, train_loader, optimizer, dfa, device)
        print(f"Backbone epoch {epoch + 1}/{epochs}: loss={loss:.4f}, train_acc={accuracy:.2f}%")
    training_seconds = time.perf_counter() - started

    representations = {}
    for split, loader in (("train", train_eval_loader), ("validation", validation_loader),
                          ("test", test_loader)):
        features, labels = collect_hidden_features(model, loader, device)
        representations[f"H_{split}"] = features.detach().cpu().float().contiguous()
        representations[f"y_{split}"] = labels.detach().cpu().long().contiguous()
    representations["training_seconds"] = training_seconds
    try:
        save_representation(path, representations, metadata)
    except FileExistsError:
        # Another process may have populated the exact requested cache after our
        # initial existence check. Validate it before reuse.
        data = load_cached()
        return data, path, "reused"
    # Round-trip through validation so newly generated artifacts and reused artifacts
    # pass the same provenance and alignment checks.
    data = load_representation(path, metadata, train_dataset, test_dataset)
    return data, path, "trained"
