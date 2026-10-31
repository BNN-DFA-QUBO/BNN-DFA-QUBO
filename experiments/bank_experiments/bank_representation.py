"""Create or safely reuse one fixed Bank BNN+DFA representation per seed/config."""
from __future__ import annotations

import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from models.bank_bnn import BankBNN
from models.bank_dfa import BankDFA, BankDFAFunction
from utils.bank.representation import (
    DEFAULT_REPRESENTATION_DIR,
    build_metadata,
    load_representation,
    representation_path,
    save_representation,
)
from utils.bank.seed import set_seed
from utils.bank.data import NUM_FEATURES, NUM_CLASSES


def collect_hidden_features(model: BankBNN, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pass loader through frozen model.fc1 + ReLU and extract hidden representations."""
    model.eval()
    all_H, all_y = [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            h = model.extract_features(features)
            all_H.append(h.cpu())
            all_y.append(labels.cpu())
    return torch.cat(all_H, dim=0), torch.cat(all_y, dim=0)


def train_bank_backbone(
    train_dataset,
    seed: int,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    hidden_size: int = 64,
    device: str | torch.device = "cpu",
) -> tuple[BankBNN, float]:
    """Train a single BNN-DFA backbone and return (model, elapsed_seconds)."""
    device = torch.device(device)
    set_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = BankBNN(input_size=NUM_FEATURES, hidden_size=hidden_size, num_classes=NUM_CLASSES).to(device)
    dfa = BankDFA(hidden_size=hidden_size, output_size=NUM_CLASSES, device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            x = features
            hidden = model.fc1(x)
            hidden_relu = torch.relu(hidden)
            outputs = model.fc2(hidden_relu)

            output_error = BankDFAFunction.output_error(outputs, labels)
            grad_output_weight = output_error.T @ hidden_relu
            grad_output_bias = output_error.sum(dim=0)

            hidden_error = dfa.hidden_gradient(output_error)
            relu_derivative = (hidden > 0).float()
            hidden_error = hidden_error * relu_derivative

            grad_hidden_weight = hidden_error.T @ x
            grad_hidden_bias = hidden_error.sum(dim=0)

            b_sz = labels.size(0)
            model.fc1.weight.grad = grad_hidden_weight / b_sz
            model.fc1.bias.grad = grad_hidden_bias / b_sz
            model.fc2.weight.grad = grad_output_weight / b_sz
            model.fc2.bias.grad = grad_output_bias / b_sz

            optimizer.step()

    elapsed = time.perf_counter() - started
    return model, elapsed


def load_or_train_bank_representation(
    train_dataset,
    val_dataset,
    test_dataset,
    seed: int,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    hidden_size: int = 64,
    device: str = "cpu",
    output_dir: Path | str = DEFAULT_REPRESENTATION_DIR,
) -> tuple[dict[str, torch.Tensor], Path, str, float, float]:
    """Load or train one fixed Bank BNN+DFA representation per seed."""
    output_dir = Path(output_dir)
    metadata = build_metadata(
        train_dataset, val_dataset, test_dataset, seed=seed, epochs=epochs,
        batch_size=batch_size, learning_rate=lr, hidden_size=hidden_size, device=device,
    )
    path = representation_path(output_dir, seed=seed, epochs=epochs)

    if path.exists():
        try:
            payload = load_representation(path, metadata, train_dataset, val_dataset, test_dataset)
            return payload, path, "reused", 0.0, 0.0
        except Exception as exc:
            print(f"Cache invalidated ({exc}), retraining fresh representation...")

    # Train fresh backbone
    dev = torch.device(device)
    model, backbone_time = train_bank_backbone(
        train_dataset, seed=seed, epochs=epochs, batch_size=batch_size, lr=lr,
        hidden_size=hidden_size, device=dev,
    )

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    # Extract hidden features across all 3 partitions
    feat_start = time.perf_counter()
    tr_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    H_train, y_train = collect_hidden_features(model, tr_loader, dev)
    H_val, y_val = collect_hidden_features(model, val_loader, dev)
    H_test, y_test = collect_hidden_features(model, te_loader, dev)
    feat_time = time.perf_counter() - feat_start

    payload = {
        "H_train": H_train,
        "y_train": y_train,
        "H_val": H_val,
        "y_val": y_val,
        "H_test": H_test,
        "y_test": y_test,
        "backbone_state": model.state_dict(),
        "backbone_training_time_sec": torch.tensor(backbone_time),
        "feature_extraction_time_sec": torch.tensor(feat_time),
    }

    save_representation(path, payload, metadata)
    return payload, path, "trained", backbone_time, feat_time

