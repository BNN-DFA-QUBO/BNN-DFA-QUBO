"""MNIST error analysis for the final BNN-DFA-QUBO model.

This script is deliberately separate from training. It expects a saved model
checkpoint and adapts to a few common checkpoint formats.

Usage examples:
    python experiments/MNIST_experiments/error_analysis.py \
        --checkpoint path/to/model.pt

If your QUBO experiment does not save a checkpoint, first add a torch.save(...)
at the end of bnn_dfa_qubo.py. See the README note below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "error_analysis"


def load_checkpoint(path):
    obj = torch.load(path, map_location="cpu")

    # Common formats:
    # 1. a complete nn.Module
    if isinstance(obj, torch.nn.Module):
        return obj

    # 2. a dict containing a complete model
    if isinstance(obj, dict):
        for key in ["model", "net", "module"]:
            if key in obj and isinstance(obj[key], torch.nn.Module):
                return obj[key]

    raise ValueError(
        "The checkpoint does not contain a complete torch.nn.Module. "
        "If your experiment saves only state_dict(), construct the exact "
        "model architecture in this script before loading it."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint(Path(args.checkpoint))
    model.eval()

    transform = transforms.ToTensor()
    test_set = datasets.MNIST(
        root=ROOT / "data",
        train=False,
        download=True,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    all_true = []
    all_pred = []
    all_images = []

    with torch.no_grad():
        for images, labels in loader:
            output = model(images)

            # Accommodate models returning tuples/lists.
            if isinstance(output, (tuple, list)):
                output = output[0]

            predictions = output.argmax(dim=1)

            all_true.extend(labels.numpy())
            all_pred.extend(predictions.numpy())
            all_images.extend(images.numpy())

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    images = np.asarray(all_images)

    # 1. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )
    ax.set_title("MNIST confusion matrix — BNN-DFA-QUBO")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Per-class accuracy
    per_class = []
    for digit in range(10):
        mask = y_true == digit
        accuracy = (y_pred[mask] == digit).mean() * 100 if mask.any() else np.nan
        per_class.append(accuracy)

    np.savetxt(
        OUT / "per_class_accuracy.csv",
        np.column_stack([np.arange(10), per_class]),
        delimiter=",",
        header="digit,accuracy_percent",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(np.arange(10), per_class)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class MNIST accuracy — BNN-DFA-QUBO")
    ax.set_xticks(np.arange(10))
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(OUT / "per_class_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Representative errors
    error_indices = np.flatnonzero(y_true != y_pred)
    n = min(20, len(error_indices))

    if n:
        fig, axes = plt.subplots(4, 5, figsize=(10, 8))
        axes = axes.ravel()

        for ax, idx in zip(axes, error_indices[:n]):
            ax.imshow(images[idx].squeeze(), cmap="gray")
            ax.set_title(f"True: {y_true[idx]}  Pred: {y_pred[idx]}")
            ax.axis("off")

        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle("Representative MNIST misclassifications")
        fig.tight_layout()
        fig.savefig(
            OUT / "misclassified_examples.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(f"Saved error-analysis results to {OUT}")


if __name__ == "__main__":
    main()
