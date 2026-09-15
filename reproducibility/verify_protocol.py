from pathlib import Path
import json
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_DIR = ROOT / "protocol"
TRAIN_PROTOCOL_DIR = PROTOCOL_DIR / "train"


def main():
    print("[VERIFY] Checking reproducibility protocol...\n")

    # ---------------------------------------------------------
    # 1. Check manifest
    # ---------------------------------------------------------
    manifest_file = PROTOCOL_DIR / "manifest.json"

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_file}"
        )

    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    print("[VERIFY] Manifest found")
    print(f"         Dataset: {manifest['dataset']}")
    print(f"         Seed: {manifest['seed']}")
    print(f"         Epochs: {manifest['num_epochs']}")
    print(f"         Train samples: {manifest['train_samples']}")
    print(f"         Test samples: {manifest['test_samples']}")

    num_epochs = manifest["num_epochs"]
    train_samples = manifest["train_samples"]
    test_samples = manifest["test_samples"]

    # ---------------------------------------------------------
    # 2. Check test indices
    # ---------------------------------------------------------
    test_file = PROTOCOL_DIR / "test_indices.npy"

    if not test_file.exists():
        raise FileNotFoundError(
            f"Missing test indices: {test_file}"
        )

    test_indices = np.load(test_file)

    print("\n[VERIFY] Test indices found")
    print(f"         Number of indices: {len(test_indices)}")

    if len(test_indices) != test_samples:
        raise ValueError(
            "Test index count does not match manifest"
        )

    expected_test_indices = np.arange(test_samples)

    if not np.array_equal(test_indices, expected_test_indices):
        raise ValueError(
            "Test indices are not the expected fixed ordering"
        )

    print("         Test ordering: OK")

    # ---------------------------------------------------------
    # 3. Check training protocol directory
    # ---------------------------------------------------------
    if not TRAIN_PROTOCOL_DIR.exists():
        raise FileNotFoundError(
            f"Missing training protocol directory: {TRAIN_PROTOCOL_DIR}"
        )

    print("\n[VERIFY] Checking training epochs...")

    all_epochs = []

    for epoch in range(num_epochs):

        epoch_file = TRAIN_PROTOCOL_DIR / f"epoch_{epoch:03d}.npy"

        if not epoch_file.exists():
            raise FileNotFoundError(
                f"Missing protocol file for epoch {epoch}: "
                f"{epoch_file}"
            )

        indices = np.load(epoch_file)

        print(
            f"         Epoch {epoch:02d}: "
            f"{len(indices)} indices",
            end=""
        )

        # Check number of samples
        if len(indices) != train_samples:
            raise ValueError(
                f"\nEpoch {epoch} contains {len(indices)} samples "
                f"instead of {train_samples}"
            )

        # Check that all indices are valid
        if np.any(indices < 0) or np.any(indices >= train_samples):
            raise ValueError(
                f"\nEpoch {epoch} contains invalid dataset indices"
            )

        # Check that every training sample appears exactly once
        sorted_indices = np.sort(indices)

        expected_indices = np.arange(train_samples)

        if not np.array_equal(sorted_indices, expected_indices):
            raise ValueError(
                f"\nEpoch {epoch} is not a valid permutation "
                f"of the training dataset"
            )

        print(" -> OK")

        all_epochs.append(indices)

    # ---------------------------------------------------------
    # 4. Check that epochs are actually different
    # ---------------------------------------------------------
    print("\n[VERIFY] Checking epoch permutations...")

    identical_pairs = []

    for i in range(len(all_epochs)):
        for j in range(i + 1, len(all_epochs)):
            if np.array_equal(all_epochs[i], all_epochs[j]):
                identical_pairs.append((i, j))

    if identical_pairs:
        print(
            "[WARNING] Some epochs use identical permutations:"
        )

        for i, j in identical_pairs:
            print(f"          Epoch {i} == Epoch {j}")

    else:
        print("         All epoch permutations are different")

    # ---------------------------------------------------------
    # 5. Final result
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("PROTOCOL VERIFICATION PASSED")
    print("=" * 60)

    print("\nThe protocol is valid and ready to be committed/used.")
    print("Do NOT regenerate it on the other platform.")
    print("Use this exact protocol on CUDA, MPS, and CPU.")


if __name__ == "__main__":
    main()