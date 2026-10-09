import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

import torch
import torch.nn as nn
import json
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score

from models.stroke_bnn import StrokeBNN
from utils.stroke_data import DATA_DIR, prepare_stroke_data
from utils.seed import set_seed
from utils.stroke_config import (
    HIDDEN_SIZE,
    BATCH_SIZE,
    TRAIN_EPOCHS,
    LEARNING_RATE
)
from utils.stroke_metrics import find_best_threshold, calculate_metrics
from utils.stroke_data import load_stroke_metadata


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


def train_bnn_bp(seed, split_seed=None, balance_method=None, overwrite=False, output_dir=DATA_DIR):
    result_path = Path(output_dir) / f"stroke_bnn_bp_seed_{seed}.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing result {result_path}; pass overwrite=True")
    set_seed(seed)

    device = get_device()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = prepare_stroke_data(split_seed=split_seed, balance_method=balance_method, overwrite=overwrite,
                            output_dir=output_dir)

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
        input_size=X_train.shape[1],
        hidden_size=HIDDEN_SIZE
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

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
        val_scores = torch.sigmoid(model(X_val_tensor)).cpu().numpy()
        test_scores = torch.sigmoid(model(torch.tensor(X_test.values, dtype=torch.float32, device=device))).cpu().numpy()
    threshold, validation_ba = find_best_threshold(y_val.values, val_scores)
    test_metrics = calculate_metrics(y_test.values, test_scores, threshold)
    metadata = load_stroke_metadata(Path(output_dir) / "metadata.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"model": "BNN backpropagation", "training_seed": seed, "split_seed": metadata["split_seed"],
                   "preprocessing_version": metadata["preprocessing_version"], "balancing_method": metadata["balancing_method"],
                   "dataset_sha256": metadata["dataset_sha256"], "preprocessing_config_hash": metadata["preprocessing_config_hash"],
                   "feature_names": metadata["feature_names"],
                   "loss": "BCEWithLogitsLoss(reduction='mean', pos_weight=None)",
                   "model_config": {"hidden_size": HIDDEN_SIZE, "batch_size": BATCH_SIZE,
                                    "epochs": TRAIN_EPOCHS, "learning_rate": LEARNING_RATE},
                   "validation_balanced_accuracy": validation_ba,
                   "split_class_counts": {"train_original": metadata["original_train_class_counts"],
                                          "train_resampled": metadata["resampled_train_class_counts"],
                                          "validation": metadata["validation_class_counts"],
                                          "test": metadata["test_class_counts"]}, **test_metrics}, f, indent=2)

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
        "y_test": y_test,
        "test_metrics": test_metrics,
        "validation_balanced_accuracy": validation_ba,
        "metrics_path": str(result_path)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the stroke BNN with backpropagation")
    parser.add_argument("--seed", type=int, default=42, help="Model training seed")
    parser.add_argument("--split-seed", type=int, default=None, help="Dataset split seed (defaults to saved/current preprocessing config)")
    parser.add_argument("--balancing", choices=("random_oversample", "none"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    args = parser.parse_args()
    result = train_bnn_bp(seed=args.seed, split_seed=args.split_seed,
                          balance_method=args.balancing, overwrite=args.overwrite, output_dir=args.output_dir)

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
    print(f"Test metrics: {result['test_metrics']}")
