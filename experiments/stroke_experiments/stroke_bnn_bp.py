import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score

from models.stroke_bnn import StrokeBNN
from utils.stroke_data import prepare_stroke_data
from utils.seed import set_seed
from utils.stroke_config import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    BATCH_SIZE,
    TRAIN_EPOCHS,
    LEARNING_RATE
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def evaluate_validation(model, X_val, y_val, device):
    model.eval()

    with torch.no_grad():
        logits = model(
            torch.tensor(
                X_val.values,
                dtype=torch.float32,
                device=device
            )
        )

    predictions = (
        torch.sigmoid(logits).cpu().numpy() >= 0.5
    ).astype(int)

    return balanced_accuracy_score(
        y_val.values,
        predictions
    )


def train_bnn_bp(seed):
    set_seed(seed)

    device = get_device()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = prepare_stroke_data()

    X_train_tensor = torch.tensor(
        X_train.values,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    model = StrokeBNN(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE
    ).to(device)

    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count

    pos_weight = torch.tensor(
        negative_count / positive_count,
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_ba = -1.0
    best_state = None
    best_epoch = 0

    X_val_tensor = torch.tensor(
        X_val.values,
        dtype=torch.float32,
        device=device
    )

    y_val_tensor = torch.tensor(
        y_val.values,
        dtype=torch.float32,
        device=device
    )

    for epoch in range(TRAIN_EPOCHS):
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()
            optimizer.step()

        val_ba = evaluate_validation(
            model,
            X_val,
            y_val,
            device
        )

        if val_ba > best_val_ba:
            best_val_ba = val_ba
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        H_train = torch.relu(
            model.fc1(
                torch.tensor(
                    X_train.values,
                    dtype=torch.float32,
                    device=device
                )
            )
        ).cpu()

        H_val = torch.relu(
            model.fc1(
                X_val_tensor
            )
        ).cpu()

        H_test = torch.relu(
            model.fc1(
                torch.tensor(
                    X_test.values,
                    dtype=torch.float32,
                    device=device
                )
            )
        ).cpu()

    return {
        "seed": seed,
        "model": model,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_ba": best_val_ba,
        "H_train": H_train,
        "H_val": H_val,
        "H_test": H_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }


if __name__ == "__main__":
    result = train_bnn_bp(seed=42)

    print(f"Device: {result['device']}")
    print(
        f"Best validation balanced accuracy: "
        f"{result['best_val_ba'] * 100:.4f}%"
    )
    print(
        f"Best epoch: {result['best_epoch']}"
    )
    print(
        f"H_train shape: {result['H_train'].shape}"
    )
    print(
        f"H_val shape: {result['H_val'].shape}"
    )
    print(
        f"H_test shape: {result['H_test'].shape}"
    )