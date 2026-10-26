"""Small serialization and provenance regression checks for experiment records."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd

from experiments.MNIST_experiments.collect_results import build_aggregate_tables
from experiments.MNIST_experiments.sweep_qubo_num_reads_shared import SELECTION_WARNING
from experiments.stroke_experiments.run_stroke_controlled import save_results
from experiments.stroke_experiments.stroke_binary_head_multiseed import write_multiseed_results
from experiments.stroke_experiments import validate_stroke_ls_heads
from utils.run_provenance import stable_run_id


class RemediationIntegrityTests(unittest.TestCase):
    def test_read_sweep_test_metrics_are_marked_descriptive_only(self):
        self.assertEqual(SELECTION_WARNING, "post_hoc_descriptive_test_metric_not_for_selection")

    def test_stroke_ls_cache_loader_checks_requested_model_configuration(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            artifact = artifact_dir / "stroke_dfa_representation_seed_42.pt"
            artifact.touch()
            data = {"split_seed": 17, "balancing_method": "none",
                    "data_metadata": {"split_seed": 17, "balancing_method": "none"}}
            args = SimpleNamespace(representation_dir=artifact_dir, split_seed=17, balancing="none")
            with mock.patch.object(validate_stroke_ls_heads, "load_representation", return_value=data) as loader:
                loaded_path, loaded, source = validate_stroke_ls_heads.load_or_train_representation(
                    42, args, root / "new",
                )
            self.assertEqual(loaded_path, artifact)
            self.assertIs(loaded, data)
            self.assertEqual(source, "reused")
            self.assertEqual(loader.call_args.kwargs["expected_seed"], 42)
            self.assertEqual(loader.call_args.kwargs["expected_hidden_size"], 64)
            self.assertEqual(loader.call_args.kwargs["expected_training_config"]["epochs"], 50)

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

    def test_multiseed_summary_groups_by_protocol_and_representation(self):
        frame = pd.DataFrame([
            {"method": "QUBO Binary", "experiment_type": "fixed", "representation_sha256": "a",
             "balanced_accuracy": 0.7, "precision": 0.6, "recall": 0.8, "f1": 0.68,
             "roc_auc": 0.75, "pr_auc": 0.7, "qubo_search_seconds": 12.5,
             "qubo_candidate_generation_seconds": 2.0, "qubo_validation_selection_seconds": 1.5,
             "qubo_num_reads_per_scale": 100, "qubo_num_sweeps": 500,
             "qubo_solver_seed_selected": 9, "validation_balanced_accuracy": 0.7,
             "threshold": 0.4, "alpha_multiplier": 1.0},
            {"method": "QUBO Binary", "experiment_type": "fixed", "representation_sha256": "b",
             "balanced_accuracy": 0.9, "precision": 0.8, "recall": 1.0, "f1": 0.88,
             "roc_auc": 0.95, "pr_auc": 0.9, "qubo_search_seconds": 13.5,
             "qubo_candidate_generation_seconds": 2.5, "qubo_validation_selection_seconds": 1.7,
             "qubo_num_reads_per_scale": 100, "qubo_num_sweeps": 500,
             "qubo_solver_seed_selected": 10, "validation_balanced_accuracy": 0.8,
             "threshold": 0.5, "alpha_multiplier": 2.0},
        ])
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            raw, summary = Path(temp) / "raw.csv", Path(temp) / "summary.csv"
            _, _, table = write_multiseed_results(raw, summary, frame.to_dict("records"))
            self.assertEqual(len(table), 2)
            saved = pd.read_csv(raw)
            for field in ("qubo_search_seconds", "qubo_candidate_generation_seconds",
                          "qubo_validation_selection_seconds", "qubo_num_reads_per_scale",
                          "qubo_num_sweeps", "qubo_solver_seed_selected", "validation_balanced_accuracy",
                          "threshold", "alpha_multiplier"):
                self.assertIn(field, saved.columns)
            self.assertEqual(saved.loc[0, "qubo_search_seconds"], 12.5)

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
