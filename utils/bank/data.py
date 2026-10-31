"""Data loading and preprocessing utilities for Bank Marketing dataset."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

NUM_FEATURES = 66
NUM_CLASSES = 2


class BankDataset(Dataset):
    """PyTorch Dataset wrapping Bank tabular features and labels."""

    def __init__(self, features: np.ndarray | torch.Tensor, targets: np.ndarray | torch.Tensor):
        if isinstance(features, np.ndarray):
            self.data = torch.from_numpy(features).float()
        else:
            self.data = features.float()

        if isinstance(targets, np.ndarray):
            self.targets = torch.from_numpy(targets).long()
        else:
            self.targets = targets.long()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def _resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is not None:
        p = Path(data_dir)
        if p.exists():
            return p
    # Search standard locations
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "data",
        root.parent / "BNN-DFA-QUBO" / "data",
        Path("data"),
    ]
    for c in candidates:
        if c.exists() and (c / "training.csv").exists():
            return c
    return root / "data"


def load_bank_split_dataframes(
    use_smote: bool = False,
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, and test dataframes with guaranteed 66 preprocessed features."""
    dir_path = _resolve_data_dir(data_dir)

    train_file = "training_smote.csv" if use_smote else "training_preprocessed.csv"
    train_path = dir_path / train_file
    val_path = dir_path / "validation_preprocessed.csv"
    test_path = dir_path / "test_preprocessed.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Preprocessed validation/test files missing in {dir_path}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_val, df_test


def get_bank_datasets(
    use_smote: bool = False,
    data_dir: str | Path | None = None,
) -> tuple[BankDataset, BankDataset, BankDataset]:
    """Return PyTorch BankDataset objects for train, val, and test splits."""
    df_train, df_val, df_test = load_bank_split_dataframes(use_smote=use_smote, data_dir=data_dir)

    feature_cols = [c for c in df_train.columns if c != "y"]
    if len(feature_cols) != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} features, got {len(feature_cols)}")

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train["y"].values.astype(np.int64)

    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val["y"].values.astype(np.int64)

    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test["y"].values.astype(np.int64)

    return BankDataset(X_train, y_train), BankDataset(X_val, y_val), BankDataset(X_test, y_test)


def get_bank_loaders(
    batch_size: int = 64,
    use_smote: bool = False,
    data_dir: str | Path | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return PyTorch DataLoaders for train, val, and test splits."""
    train_ds, val_ds, test_ds = get_bank_datasets(use_smote=use_smote, data_dir=data_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

