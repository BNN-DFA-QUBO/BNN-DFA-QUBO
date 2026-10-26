"""Result schema compatibility tests, independent of MNIST data."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from experiments.MNIST_experiments import results_utils
from experiments.MNIST_experiments.run_mnist_shared_validation import _persist_selected_qubo_result


class MnistResultsSchemaTests(unittest.TestCase):
    def test_shared_validation_persists_one_selected_qubo_row_with_test_metrics(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "mnist_results.csv"
            tuning_csv = Path(temp) / "qubo_validation_tuning_27config_seed_42.csv"
            tuning_csv.write_text("synthetic tuning artifact\n", encoding="utf-8")
            representation_path = Path(temp) / "representation.pt"
            representation = {
                "artifact_id": "artifact-42",
                "representation_sha256": "sha-42",
                "metadata": {
                    "backbone_parameter_count": 12345,
                    "sample_counts": {"train": 51000, "validation": 9000, "test": 10000},
                    "dataset_fingerprint": "dataset-42", "split_protocol": "stratified-v1",
                    "training": {"epochs": 16, "seed": 42},
                },
            }
            winner = {
                "num_reads": 50,
                "num_sweeps": 500,
                "scale_multiplier": 2.0,
                "validation_accuracy": 88.75,
                "solver_seed": 1051,
            }
            tuning_rows = [{"runtime_sec": 0.25} for _ in range(27)]

            with mock.patch.object(results_utils, "RESULTS_CSV", path), \
                 mock.patch.object(results_utils, "get_device_name", return_value="cpu"), \
                 mock.patch.object(results_utils, "get_hardware_label", return_value="test"):
                _persist_selected_qubo_result(
                    seed=42,
                    test_accuracy=91.25,
                    winner=winner,
                    tuning_rows=tuning_rows,
                    validation_fraction=0.15,
                    representation=representation,
                    representation_path=representation_path,
                    representation_source="reused",
                    tuning_csv_path=tuning_csv,
                )

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["experiment"], "mnist_shared_validation")
            self.assertEqual(row["method"], "QUBO Binary")
            self.assertEqual(row["seed"], "42")
            self.assertEqual(row["test_accuracy"], "91.25")
            self.assertEqual(row["qubo_num_reads_selected"], "50")
            self.assertEqual(row["qubo_num_sweeps_selected"], "500")
            self.assertEqual(row["qubo_scale_multiplier_selected"], "2.0")
            self.assertEqual(row["qubo_validation_accuracy"], "88.75")
            self.assertEqual(row["qubo_solver_seed_selected"], "1051")
            self.assertEqual(row["qubo_validation_fraction"], "0.15")
            self.assertEqual(row["qubo_grid_size"], "27")
            self.assertEqual(row["qubo_grid_protocol"], "fixed_27_configuration_v1")
            self.assertEqual(row["representation_artifact_id"], "artifact-42")
            self.assertEqual(row["representation_sha256"], "sha-42")
            self.assertEqual(row["dataset_fingerprint"], "dataset-42")
            self.assertEqual(row["split_protocol"], "stratified-v1")
            self.assertEqual(row["n_train"], "51000")
            self.assertEqual(row["n_validation"], "9000")
            self.assertEqual(row["n_test"], "10000")
            self.assertIn(tuning_csv.name, row["notes"])

    def test_tuning_fields_append_without_losing_historical_columns_or_rows(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "mnist_results.csv"
            path.write_text("experiment,method,seed,test_accuracy,legacy_extra\n"
                            "old,baseline,1,90.0,preserved\n", encoding="utf-8")
            with mock.patch.object(results_utils, "RESULTS_CSV", path), \
                 mock.patch.object(results_utils, "get_device_name", return_value="cpu"), \
                 mock.patch.object(results_utils, "get_hardware_label", return_value="test"):
                results_utils.append_result(
                    experiment="bnn_dfa_qubo", method="tuned", seed=42,
                    test_accuracy=91.0, qubo_tuning_enabled=True,
                    qubo_num_reads_selected=50, qubo_scale_multiplier_selected=1.0,
                    qubo_num_sweeps_selected=500,
                    qubo_grid_protocol="fixed_27_configuration_v1",
                    qubo_validation_accuracy=88.5, qubo_validation_fraction=0.15,
                    qubo_grid_size=9, qubo_solver_seed_selected=1051,
                    qubo_selection_metric="validation_accuracy_percent",
                    representation_artifact_id="artifact123",
                    representation_path="data/mnist_representations/test.pt",
                    representation_cache_status="reused",
                    n_train=51000, n_validation=9000, n_test=10000,
                )
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                columns = reader.fieldnames
            self.assertIn("legacy_extra", columns)
            self.assertIn("qubo_validation_accuracy", columns)
            self.assertEqual(rows[0]["legacy_extra"], "preserved")
            self.assertEqual(rows[0]["qubo_num_reads_selected"], "")
            self.assertEqual(rows[1]["qubo_num_reads_selected"], "50")
            self.assertEqual(rows[1]["qubo_solver_seed_selected"], "1051")
            self.assertEqual(rows[1]["qubo_selection_metric"], "validation_accuracy_percent")
            self.assertEqual(rows[1]["representation_artifact_id"], "artifact123")
            self.assertEqual(rows[1]["n_train"], "51000")
            self.assertEqual(rows[1]["qubo_num_sweeps_selected"], "500")
            self.assertEqual(rows[1]["qubo_grid_protocol"], "fixed_27_configuration_v1")


if __name__ == "__main__":
    unittest.main()
