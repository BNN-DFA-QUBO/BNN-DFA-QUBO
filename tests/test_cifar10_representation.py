"""Unit tests for CIFAR-10 representation metadata, hashing, caching, and integrity."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from tests.test_cifar10_data import MockCIFAR10Dataset
from utils.CIFAR10.representation import (
    build_metadata,
    dataset_fingerprint,
    load_representation,
    representation_path,
    save_representation,
)


class TestCIFAR10Representation(unittest.TestCase):

    def setUp(self):
        self.train_ds = MockCIFAR10Dataset(num_samples=100)
        self.test_ds = MockCIFAR10Dataset(num_samples=20)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_fingerprint_deterministic(self):
        fp1 = dataset_fingerprint(self.train_ds, self.test_ds)
        fp2 = dataset_fingerprint(self.train_ds, self.test_ds)
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64, "Fingerprint must be a 64-char hex SHA-256 digest")

    def test_save_and_load_roundtrip_and_cache_reuse(self):
        seed = 42
        epochs = 2
        val_frac = 0.2
        metadata = build_metadata(
            self.train_ds, self.test_ds, seed=seed, epochs=epochs, validation_fraction=val_frac
        )
        path = representation_path(self.out_dir, seed=seed, epochs=epochs, validation_fraction=val_frac)

        indices = metadata["split_indices"]
        n_tr, n_val, n_te = len(indices["train"]), len(indices["validation"]), len(indices["test"])

        data = {
            "H_train": torch.randn(n_tr, 128, dtype=torch.float32),
            "y_train": torch.as_tensor([self.train_ds.targets[i] for i in indices["train"]], dtype=torch.long),
            "H_validation": torch.randn(n_val, 128, dtype=torch.float32),
            "y_validation": torch.as_tensor([self.train_ds.targets[i] for i in indices["validation"]], dtype=torch.long),
            "H_test": torch.randn(n_te, 128, dtype=torch.float32),
            "y_test": torch.as_tensor([self.test_ds.targets[i] for i in indices["test"]], dtype=torch.long),
        }

        # Save representation
        saved = save_representation(path, data, metadata)
        self.assertTrue(path.exists())
        self.assertIn("artifact_id", saved)
        self.assertIn("representation_sha256", saved)

        # Load representation
        loaded = load_representation(path, metadata, self.train_ds, self.test_ds)
        self.assertEqual(loaded["artifact_id"], saved["artifact_id"])
        self.assertTrue(torch.equal(loaded["H_train"], data["H_train"]))
        self.assertTrue(torch.equal(loaded["y_train"], data["y_train"]))

    def test_overwrite_protection(self):
        seed = 42
        epochs = 2
        val_frac = 0.2
        metadata = build_metadata(
            self.train_ds, self.test_ds, seed=seed, epochs=epochs, validation_fraction=val_frac
        )
        path = representation_path(self.out_dir, seed=seed, epochs=epochs, validation_fraction=val_frac)
        indices = metadata["split_indices"]
        data = {
            "H_train": torch.randn(len(indices["train"]), 128),
            "y_train": torch.as_tensor([self.train_ds.targets[i] for i in indices["train"]], dtype=torch.long),
            "H_validation": torch.randn(len(indices["validation"]), 128),
            "y_validation": torch.as_tensor([self.train_ds.targets[i] for i in indices["validation"]], dtype=torch.long),
            "H_test": torch.randn(len(indices["test"]), 128),
            "y_test": torch.as_tensor([self.test_ds.targets[i] for i in indices["test"]], dtype=torch.long),
        }

        save_representation(path, data, metadata)
        # Second save must raise FileExistsError
        with self.assertRaises(FileExistsError):
            save_representation(path, data, metadata)

    def test_reject_incompatible_metadata(self):
        seed = 42
        epochs = 2
        val_frac = 0.2
        metadata = build_metadata(
            self.train_ds, self.test_ds, seed=seed, epochs=epochs, validation_fraction=val_frac
        )
        path = representation_path(self.out_dir, seed=seed, epochs=epochs, validation_fraction=val_frac)
        indices = metadata["split_indices"]
        data = {
            "H_train": torch.randn(len(indices["train"]), 128),
            "y_train": torch.as_tensor([self.train_ds.targets[i] for i in indices["train"]], dtype=torch.long),
            "H_validation": torch.randn(len(indices["validation"]), 128),
            "y_validation": torch.as_tensor([self.train_ds.targets[i] for i in indices["validation"]], dtype=torch.long),
            "H_test": torch.randn(len(indices["test"]), 128),
            "y_test": torch.as_tensor([self.test_ds.targets[i] for i in indices["test"]], dtype=torch.long),
        }
        save_representation(path, data, metadata)

        # Incompatible metadata (e.g. wrong epochs or seed)
        incompatible_meta = dict(metadata)
        incompatible_meta["training"] = dict(metadata["training"])
        incompatible_meta["training"]["epochs"] = 99

        with self.assertRaises(ValueError) as ctx:
            load_representation(path, incompatible_meta, self.train_ds, self.test_ds)
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_reject_corrupted_artifact(self):
        seed = 42
        epochs = 2
        val_frac = 0.2
        metadata = build_metadata(
            self.train_ds, self.test_ds, seed=seed, epochs=epochs, validation_fraction=val_frac
        )
        path = representation_path(self.out_dir, seed=seed, epochs=epochs, validation_fraction=val_frac)
        indices = metadata["split_indices"]
        data = {
            "H_train": torch.randn(len(indices["train"]), 128),
            "y_train": torch.as_tensor([self.train_ds.targets[i] for i in indices["train"]], dtype=torch.long),
            "H_validation": torch.randn(len(indices["validation"]), 128),
            "y_validation": torch.as_tensor([self.train_ds.targets[i] for i in indices["validation"]], dtype=torch.long),
            "H_test": torch.randn(len(indices["test"]), 128),
            "y_test": torch.as_tensor([self.test_ds.targets[i] for i in indices["test"]], dtype=torch.long),
        }
        saved = save_representation(path, data, metadata)

        # Corrupt data on disk (tamper with a tensor without updating hash)
        corrupted = dict(saved)
        corrupted["H_train"][0, 0] += 10.0
        torch.save(corrupted, path)

        with self.assertRaises(ValueError) as ctx:
            load_representation(path, metadata, self.train_ds, self.test_ds)
        self.assertIn("checksum is invalid", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
