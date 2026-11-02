"""Tests for MNIST shared validation results logging and deduplication."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.MNIST_experiments.results_utils import append_result
from experiments.MNIST_experiments.run_mnist_shared_validation import _persist_all_shared_results


class TestMnistSharedValidationPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmp_dir.name) / "test_mnist_results.csv"

    def tearDown(self):
        self.tmp_dir.cleanup()

    @mock.patch("experiments.MNIST_experiments.results_utils.RESULTS_CSV")
    def test_persist_all_shared_results_records_all_evaluated_heads(self, mock_results_csv):
        mock_results_csv.parent = self.csv_path.parent
        mock_results_csv.open = self.csv_path.open
        mock_results_csv.exists = self.csv_path.exists

        rows = [
            {"head": "Ridge LS", "test_accuracy_pct": 94.27, "head_fit_time_sec": 0.05, "head_parameter_count": 1290},
            {"head": "Binarized Ridge LS", "test_accuracy_pct": 70.54, "head_fit_time_sec": 0.08, "head_parameter_count": 1300},
            {"head": "QUBO Binary", "test_accuracy_pct": 93.23, "head_fit_time_sec": 1.20, "head_parameter_count": 1300},
            {"head": "STE (best validation epoch)", "test_accuracy_pct": 91.43, "head_fit_time_sec": 0.45, "head_parameter_count": 1290},
        ]
        winner = {
            "num_reads": 25,
            "num_sweeps": 250,
            "scale_multiplier": 1.0,
            "validation_accuracy": 93.00,
            "solver_seed": 42,
            "runtime_sec": 0.05,
        }
        tuning_rows = [winner]
        representation = {
            "artifact_id": "art_test_123",
            "metadata": {
                "sample_counts": {"train": 54000, "validation": 6000, "test": 10000},
                "dataset_fingerprint": "fp_test",
                "split_protocol": "mnist_train_val_test",
                "training": {},
            },
            "representation_sha256": "sha_test_123",
        }

        _persist_all_shared_results(
            seed=42,
            rows=rows,
            winner=winner,
            tuning_rows=tuning_rows,
            validation_fraction=0.1,
            representation=representation,
            representation_path=Path("fake_rep.pt"),
            representation_source="reused",
            tuning_csv_path=Path("fake_tuning.csv"),
            backbone_training_time_sec=10.0,
            feature_extraction_time_sec=2.0,
            cur_backbone_time=0.0,
            cur_extraction_time=0.0,
            pipeline_timing_status="complete",
            backbone_parameter_count=101770,
        )

        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            logged = list(reader)

        self.assertEqual(len(logged), 4)
        methods = [r["method"] for r in logged]
        self.assertIn("Ridge LS", methods)
        self.assertIn("Binarized Ridge LS", methods)
        self.assertIn("QUBO Binary", methods)
        self.assertIn("STE Binary", methods)

        for r in logged:
            self.assertEqual(r["experiment"], "mnist_shared_validation")
            self.assertEqual(r["seed"], "42")

        # Calling again should safely skip duplicate append without raising FileExistsError
        _persist_all_shared_results(
            seed=42,
            rows=rows,
            winner=winner,
            tuning_rows=tuning_rows,
            validation_fraction=0.1,
            representation=representation,
            representation_path=Path("fake_rep.pt"),
            representation_source="reused",
            tuning_csv_path=Path("fake_tuning.csv"),
            backbone_training_time_sec=10.0,
            feature_extraction_time_sec=2.0,
            cur_backbone_time=0.0,
            cur_extraction_time=0.0,
            pipeline_timing_status="complete",
            backbone_parameter_count=101770,
        )

        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            logged_again = list(reader)

        self.assertEqual(len(logged_again), 4)


if __name__ == "__main__":
    unittest.main()

