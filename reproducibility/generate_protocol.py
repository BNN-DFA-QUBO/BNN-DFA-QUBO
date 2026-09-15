from pathlib import Path
import json

import numpy as np
from torchvision import datasets


SEED = 42
NUM_EPOCHS = 16

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
PROTOCOL_DIR = ROOT / "protocol"
TRAIN_PROTOCOL_DIR = PROTOCOL_DIR / "train"


def main():

    PROTOCOL_DIR.mkdir(exist_ok=True)
    TRAIN_PROTOCOL_DIR.mkdir(exist_ok=True)

    print("[REPRO] Loading MNIST...")

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True
    )

    print(f"[REPRO] Training samples: {len(train_dataset)}")
    print(f"[REPRO] Test samples: {len(test_dataset)}")

    # ---------------------------------------------------------
    # TEST DATA
    # ---------------------------------------------------------

    # MNIST's test set already has a fixed ordering.
    test_indices = np.arange(len(test_dataset))

    np.save(
        PROTOCOL_DIR / "test_indices.npy",
        test_indices
    )

    # ---------------------------------------------------------
    # TRAINING ORDER
    # ---------------------------------------------------------

    # One deterministic RNG.
    rng = np.random.default_rng(SEED)

    for epoch in range(NUM_EPOCHS):

        # Generate a predetermined random permutation.
        indices = rng.permutation(len(train_dataset))

        output_file = (
            TRAIN_PROTOCOL_DIR /
            f"epoch_{epoch:03d}.npy"
        )

        np.save(output_file, indices)

        print(
            f"[REPRO] Epoch {epoch}: "
            f"saved {len(indices)} training indices"
        )

    # ---------------------------------------------------------
    # MANIFEST
    # ---------------------------------------------------------

    manifest = {
        "dataset": "MNIST",
        "seed": SEED,
        "num_epochs": NUM_EPOCHS,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset)
    }

    with open(
        PROTOCOL_DIR / "manifest.json",
        "w"
    ) as f:
        json.dump(manifest, f, indent=4)

    print()
    print("[REPRO] =================================")
    print("[REPRO] Protocol generated successfully")
    print("[REPRO] =================================")
    print()
    print(f"[REPRO] Location: {PROTOCOL_DIR}")


if __name__ == "__main__":
    main()