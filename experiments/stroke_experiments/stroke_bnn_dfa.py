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
from models.stroke_dfa import StrokeDFA, StrokeDFAFunction
from utils.stroke_data import DATA_DIR, prepare_stroke_data
from utils.stroke_representation import save_representation
from utils.stroke_data import load_stroke_metadata
from utils.stroke_metrics import find_best_threshold, calculate_metrics
from utils.seed import set_seed
from utils.stroke_config import (
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


def train_dfa(seed, split_seed=None, balance_method=None, overwrite=False, output_dir=DATA_DIR):
    output_dir = Path(output_dir)
    result_path = output_dir / f"stroke_bnn_dfa_seed_{seed}.json"
    representation_path = output_dir / f"stroke_dfa_representation_seed_{seed}.pt"
    existing = [path for path in (result_path, representation_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to replace existing stroke outputs: " + ", ".join(existing) + "; pass overwrite=True")
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

    dfa = StrokeDFA(
        hidden_size=HIDDEN_SIZE,
        device=device
    )

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

    for epoch in range(TRAIN_EPOCHS):
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            hidden_pre = model.fc1(X_batch)
            hidden = torch.relu(hidden_pre)
            logits = model.fc2(hidden).squeeze(1)

            output_error = StrokeDFAFunction.output_error(
                logits,
                y_batch
            )

            output_error = (
                output_error / y_batch.size(0)
            )

            output_weight_gradient = (
                output_error.unsqueeze(1) * hidden
            ).sum(dim=0, keepdim=True)

            output_bias_gradient = (
                output_error.sum()
            )

            hidden_gradient = dfa.hidden_gradient(
                output_error.unsqueeze(1)
            )

            hidden_gradient = (
                hidden_gradient *
                (hidden_pre > 0).float()
            )

            hidden_weight_gradient = (
                hidden_gradient.unsqueeze(2) *
                X_batch.unsqueeze(1)
            ).sum(dim=0)

            hidden_bias_gradient = (
                hidden_gradient.sum(dim=0)
            )

            model.fc1.weight.grad = (
                hidden_weight_gradient
            )

            model.fc1.bias.grad = (
                hidden_bias_gradient
            )

            model.fc2.weight.grad = (
                output_weight_gradient
            )

            model.fc2.bias.grad = (
                output_bias_gradient.unsqueeze(0)
            )

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
    metadata = load_stroke_metadata(output_dir / "metadata.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"model": "BNN DFA output", "training_seed": seed, "split_seed": metadata["split_seed"],
                   "preprocessing_version": metadata["preprocessing_version"], "balancing_method": metadata["balancing_method"],
                   "dataset_sha256": metadata["dataset_sha256"], "preprocessing_config_hash": metadata["preprocessing_config_hash"],
                   "feature_names": metadata["feature_names"],
                   "loss": "BCEWithLogitsLoss(reduction='mean', pos_weight=None)",
                   "model_config": {"hidden_size": HIDDEN_SIZE, "batch_size": BATCH_SIZE,
                                    "epochs": TRAIN_EPOCHS, "learning_rate": LEARNING_RATE},
                   "threshold": threshold,
                   "validation_balanced_accuracy": validation_ba,
                   "split_class_counts": {"train_original": metadata["original_train_class_counts"],
                                          "train_resampled": metadata["resampled_train_class_counts"],
                                          "validation": metadata["validation_class_counts"],
                                          "test": metadata["test_class_counts"]}, **test_metrics}, f, indent=2)

    with torch.no_grad():
        X_train_device = torch.tensor(
            X_train.values,
            dtype=torch.float32,
            device=device
        )

        X_test_device = torch.tensor(
            X_test.values,
            dtype=torch.float32,
            device=device
        )

        H_train = torch.relu(
            model.fc1(X_train_device)
        ).cpu()

        H_val = torch.relu(
            model.fc1(X_val_tensor)
        ).cpu()

        H_test = torch.relu(
            model.fc1(X_test_device)
        ).cpu()

    return {
        "seed": seed,
        "model": model,
        "dfa_feedback": dfa.feedback.detach().cpu(),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_ba": best_val_ba,
        "H_train": H_train,
        "H_val": H_val,
        "H_test": H_test,
        "feature_names": list(X_train.columns),
        "data_metadata": load_stroke_metadata(output_dir / "metadata.json"),
        "training_seed": seed,
        "training_config": {"hidden_size": HIDDEN_SIZE, "batch_size": BATCH_SIZE,
                            "epochs": TRAIN_EPOCHS, "learning_rate": LEARNING_RATE,
                            "loss": "BCEWithLogitsLoss(reduction='mean', pos_weight=None)"},
        "test_metrics": test_metrics,
        "validation_balanced_accuracy": validation_ba,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the stroke BNN using DFA and save hidden representations")
    parser.add_argument("--seed", type=int, default=42, help="Model training seed")
    parser.add_argument("--split-seed", type=int, default=None, help="Dataset split seed (defaults to saved/current preprocessing config)")
    parser.add_argument("--balancing", choices=("random_oversample", "none"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    args = parser.parse_args()
    seed = args.seed

    result = train_dfa(seed, split_seed=args.split_seed, balance_method=args.balancing,
                       overwrite=args.overwrite, output_dir=args.output_dir)

    output_path = (
        str(Path(args.output_dir) / f"stroke_dfa_representation_seed_{seed}.pt")
    )

    save_representation(
        result,
        output_path,
        feature_names=result["feature_names"]
    )

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
    print(
        f"DFA feedback shape: "
        f"{result['dfa_feedback'].shape}"
    )
    print(
        f"Representation saved to: {output_path}"
    )
