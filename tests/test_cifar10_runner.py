"""Unit tests for the CIFAR-10 shared-representation runner and multi-seed controlled pipeline."""
import os
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import torch

from experiments.CIFAR10_experiments.results_utils import (
    FIELDNAMES,
    append_result,
)
from experiments.CIFAR10_experiments.run_cifar10_shared_validation import (
    run_cifar10_shared_validation,
)
from tests.test_cifar10_data import MockCIFAR10Dataset
from utils.CIFAR10.representation import (
    build_metadata,
    representation_path,
    save_representation,
)


class TestCIFAR10Runner(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.rep_dir = self.root / "cifar10_representations"
        self.results_csv = self.root / "cifar10_results.csv"

        self.mnist_csv = self.root / "mnist_results.csv"
        # Pre-seed mnist_csv with placeholder to verify it is untouched
        self.mnist_csv.write_text("dummy_mnist_content\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_shared_representation_consistency_across_heads(self):
        # Create a pre-cached synthetic representation artifact
        seed = 42
        epochs = 1
        val_frac = 0.2
        train_ds = MockCIFAR10Dataset(num_samples=50)
        test_ds = MockCIFAR10Dataset(num_samples=20)

        metadata = build_metadata(
            train_ds, test_ds, seed=seed, epochs=epochs, validation_fraction=val_frac
        )
        path = representation_path(self.rep_dir, seed=seed, epochs=epochs, validation_fraction=val_frac)
        indices = metadata["split_indices"]

        data = {
            "H_train": torch.randn(len(indices["train"]), 128),
            "y_train": torch.as_tensor([train_ds.targets[i] for i in indices["train"]], dtype=torch.long),
            "H_validation": torch.randn(len(indices["validation"]), 128),
            "y_validation": torch.as_tensor([train_ds.targets[i] for i in indices["validation"]], dtype=torch.long),
            "H_test": torch.randn(len(indices["test"]), 128),
            "y_test": torch.as_tensor([test_ds.targets[i] for i in indices["test"]], dtype=torch.long),
            "training_seconds": 1.5,
            "feature_extraction_seconds": 0.5,
        }
        saved = save_representation(path, data, metadata)
        expected_artifact_id = saved["artifact_id"]

        os.environ["CIFAR10_RESULTS_CSV"] = str(self.results_csv)
        try:
            results = run_cifar10_shared_validation(
                seed=seed,
                epochs=epochs,
                ste_epochs=5,
                direct_steps=10,
                validation_fraction=val_frac,
                representation_dir=self.rep_dir,
                fixed_qubo=True,
                num_reads=5,
                num_sweeps=10,
                device="cpu",
                train_dataset=train_ds,
                test_dataset=test_ds,
            )
            self.assertEqual(len(results), 5, "Must evaluate all 5 heads")

            # Check that results CSV was written
            self.assertTrue(self.results_csv.exists())
            df = pd.read_csv(self.results_csv)
            self.assertEqual(len(df), 5)

            # Check all 5 records share the exact same representation artifact ID
            artifact_ids = df["representation_artifact_id"].tolist()
            self.assertEqual(set(artifact_ids), {expected_artifact_id})

            # Verify MNIST results CSV is untouched
            self.assertEqual(
                self.mnist_csv.read_text(encoding="utf-8"),
                "dummy_mnist_content\n",
                "MNIST results file must NEVER be modified by CIFAR experiments",
            )

        finally:
            os.environ.pop("CIFAR10_RESULTS_CSV", None)

    def test_results_csv_schema_and_integrity(self):
        append_result(
            experiment="cifar_test_exp",
            method="Test Method",
            seed=42,
            test_accuracy=85.5,
            validation_accuracy=86.0,
            training_accuracy=90.0,
            results_path=self.results_csv,
        )
        self.assertTrue(self.results_csv.exists())
        df = pd.read_csv(self.results_csv)
        self.assertEqual(len(df), 1)

        for field in FIELDNAMES:
            self.assertIn(field, df.columns)

        row = df.iloc[0]
        self.assertEqual(row["dataset"], "CIFAR-10")
        self.assertEqual(row["method"], "Test Method")
        self.assertAlmostEqual(float(row["test_accuracy"]), 85.5)


if __name__ == "__main__":
    unittest.main()
