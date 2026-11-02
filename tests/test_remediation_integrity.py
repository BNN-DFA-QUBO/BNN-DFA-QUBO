"""Small serialization and provenance regression checks for experiment records."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from experiments.MNIST_experiments.collect_results import build_aggregate_tables
from experiments.stroke_experiments.run_stroke_controlled import save_results
from utils.run_provenance import stable_run_id


class RemediationIntegrityTests(unittest.TestCase):
    def test_controlled_stroke_writer_persists_representation_and_provenance_fields(self):
        row = {
            "experiment_type": "controlled_end_to_end_head_comparison",
            "method": "QUBO Binary", "seed": 42, "status": "ok",
            "representation_path": "representation.pt", "representation_sha256": "abc123",
            "run_id": "stroke_42", "qubo_search_seconds": 12.5,
            "qubo_num_reads": 100, "qubo_num_sweeps": 200,
            "git_commit_sha": "deadbeef", "provenance_json": "{}",
        }
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "stroke.csv"
            save_results([row], path)
            with path.open(newline="", encoding="utf-8") as handle:
                saved = next(csv.DictReader(handle))
            self.assertEqual(saved["representation_sha256"], "abc123")
            self.assertEqual(saved["run_id"], "stroke_42")
            self.assertEqual(saved["qubo_search_seconds"], "12.5")
            self.assertEqual(saved["qubo_num_reads"], "100")
            self.assertEqual(saved["git_commit_sha"], "deadbeef")
            with self.assertRaises(FileExistsError):
                save_results([row], path)

    def test_mnist_aggregates_separate_protocols_and_selected_configs(self):
        frame = pd.DataFrame([
            {"experiment": "mnist", "method": "QUBO", "experiment_type": "shared",
             "qubo_grid_protocol": "fixed27", "qubo_num_reads_selected": 25,
             "test_accuracy": 90.0, "training_time_sec": 1.0, "parameter_count": 10,
             "qubo_time_sec": 2.0},
            {"experiment": "mnist", "method": "QUBO", "experiment_type": "historical",
             "qubo_grid_protocol": "fixed", "qubo_num_reads_selected": 100,
             "test_accuracy": 95.0, "training_time_sec": 3.0, "parameter_count": 10,
             "qubo_time_sec": 4.0},
        ])
        tables = build_aggregate_tables(frame)
        main = tables["main_classification_results.csv"]
        self.assertEqual(len(main), 2)
        self.assertEqual(set(main["runs"]), {1})

    def test_run_id_is_stable_for_same_protocol_configuration_and_provenance(self):
        provenance = {"git_commit_sha": "abc", "git_worktree_fingerprint": "def"}
        config = {"seed": 42, "split": "train-val-test"}
        self.assertEqual(stable_run_id("test", config, provenance),
                         stable_run_id("test", config, provenance))
        self.assertNotEqual(stable_run_id("test", config, provenance),
                             stable_run_id("test", {"seed": 43}, provenance))


if __name__ == "__main__":
    unittest.main()
