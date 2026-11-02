"""Unit and regression tests for Stroke shared-backbone experiment and QUBO tuning."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from experiments.stroke_experiments.tune_qubo_validation import (
    DEFAULT_NUM_READS,
    DEFAULT_NUM_SWEEPS,
    DEFAULT_SCALE_MULTIPLIERS,
    validate_grid,
    select_configuration,
    tune_from_representations,
    write_tuning_csv,
)
from experiments.stroke_experiments.run_stroke_shared_validation import (
    run_stroke_shared_validation,
    _save_shared_results_csv,
    _append_canonical_results,
)


class TestStrokeQuboTuning(unittest.TestCase):
    def test_validate_grid_defaults(self):
        grid = validate_grid(DEFAULT_NUM_READS, DEFAULT_NUM_SWEEPS, DEFAULT_SCALE_MULTIPLIERS)
        self.assertEqual(len(grid), 27)
        self.assertEqual(grid[0], (25, 250, 0.5))

    def test_validate_grid_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            validate_grid((0,), (250,), (1.0,))
        with self.assertRaises(ValueError):
            validate_grid((25,), (-10,), (1.0,))
        with self.assertRaises(ValueError):
            validate_grid((25,), (250,), (-1.0,))
        with self.assertRaises(ValueError):
            validate_grid((25, 25), (250,), (1.0,))

    def test_select_configuration_deterministic_tie_breaking(self):
        rows = [
            {
                "status": "ok",
                "validation_selection_score": 0.80,
                "validation_balanced_accuracy": 0.80,
                "num_reads": 100,
                "scale_multiplier": 1.0,
                "num_sweeps": 500,
            },
            {
                "status": "ok",
                "validation_selection_score": 0.80,
                "validation_balanced_accuracy": 0.80,
                "num_reads": 25,
                "scale_multiplier": 1.0,
                "num_sweeps": 500,
            },
            {
                "status": "ok",
                "validation_selection_score": 0.75,
                "validation_balanced_accuracy": 0.75,
                "num_reads": 10,
                "scale_multiplier": 1.0,
                "num_sweeps": 250,
            },
            {
                "status": "failed",
                "validation_selection_score": 0.99,
                "validation_balanced_accuracy": 0.99,
                "num_reads": 10,
                "scale_multiplier": 1.0,
                "num_sweeps": 250,
            },
        ]
        winner = select_configuration(rows)
        # Winner must be the second row (highest score, fewest reads)
        self.assertEqual(winner["num_reads"], 25)
        self.assertEqual(winner["validation_selection_score"], 0.80)

    def test_write_tuning_csv_has_no_test_columns(self):
        rows = [
            {
                "status": "ok",
                "num_reads": 25,
                "num_sweeps": 250,
                "scale_multiplier": 1.0,
                "solver_seed": 1001,
                "validation_selection_score": 0.75,
                "validation_balanced_accuracy": 0.75,
                "validation_f1": 0.70,
                "validation_pr_auc": 0.72,
                "validation_roc_auc": 0.78,
                "threshold": 0.1,
                "alpha": 0.05,
                "base_alpha": 0.05,
                "bias": -0.01,
                "energy": -12.3,
                "candidate_source": "annealing",
                "candidate_count": 5,
                "selected": True,
                "error": "",
                "runtime_sec": 1.5,
                "tuning_total_seconds": 15.0,
                "selected_candidate_runtime_sec": 1.5,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = write_tuning_csv(
                rows=rows,
                seed=42,
                output_dir=tmp_dir,
                artifact_id="art123",
                n_train=100,
                n_val=20,
                n_test=20,
            )
            self.assertTrue(csv_path.exists())
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames
                self.assertNotIn("test_accuracy", header)
                self.assertNotIn("balanced_accuracy", header)
                self.assertIn("validation_balanced_accuracy", header)
                self.assertIn("validation_selection_score", header)
                self.assertIn("num_reads", header)
                self.assertIn("scale_multiplier", header)


class TestStrokeSharedValidation(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.H_train = torch.randn(40, 64)
        self.H_val = torch.randn(10, 64)
        self.H_test = torch.randn(10, 64)
        self.y_train = torch.tensor([0.0, 1.0] * 20)
        self.y_val = torch.tensor([0.0, 1.0] * 5)
        self.y_test = torch.tensor([0.0, 1.0] * 5)

        self.mock_rep = {
            "H_train": self.H_train,
            "H_val": self.H_val,
            "H_test": self.H_test,
            "y_train": self.y_train,
            "y_val": self.y_val,
            "y_test": self.y_test,
            "training_seed": self.seed,
            "representation_sha256": "fake_hash_123",
            "training_seconds": 12.0,
            "feature_extraction_seconds": 0.5,
            "feature_names": [f"f{i}" for i in range(15)],
            "data_metadata": {
                "split_seed": 42,
                "preprocessing_version": "v3",
                "preprocessing_config_hash": "cfg123",
                "dataset_sha256": "data123",
                "balancing_method": "none",
            },
        }
        self.mock_metadata = {
            "split_seed": 42,
            "preprocessing_version": "v3",
            "preprocessing_config_hash": "cfg123",
            "dataset_sha256": "data123",
            "balancing_method": "none",
        }

    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.prepare_stroke_data")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.load_stroke_metadata")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.load_or_train_stroke_representation")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.train_real_ls")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.train_binarized_ls")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.train_direct_binary")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.train_ste_binary")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.train_qubo")
    def test_all_heads_evaluated_on_shared_representation_fixed_qubo(
        self,
        mock_qubo,
        mock_ste,
        mock_direct,
        mock_bin_ridge,
        mock_real_ls,
        mock_load_rep,
        mock_meta,
        mock_prep,
    ):
        mock_meta.return_value = self.mock_metadata
        mock_load_rep.return_value = (self.mock_rep, Path("fake_rep.pt"), "reused")

        mock_real_ls.return_value = {"balanced_accuracy": 0.70, "f1": 0.20, "roc_auc": 0.75, "pr_auc": 0.15, "threshold": 0.5}
        mock_bin_ridge.return_value = {"balanced_accuracy": 0.71, "f1": 0.21, "roc_auc": 0.76, "pr_auc": 0.16, "threshold": 0.5}
        mock_direct.return_value = {"balanced_accuracy": 0.69, "f1": 0.19, "roc_auc": 0.74, "pr_auc": 0.14, "threshold": 0.0}
        mock_ste.return_value = {"balanced_accuracy": 0.72, "f1": 0.22, "roc_auc": 0.77, "pr_auc": 0.17, "threshold": 0.5, "initialization": "class_mean"}
        mock_qubo.return_value = {
            "balanced_accuracy": 0.73,
            "f1": 0.23,
            "roc_auc": 0.78,
            "pr_auc": 0.18,
            "threshold": 0.5,
            "validation_balanced_accuracy": 0.75,
            "qubo_search_seconds": 2.0,
            "qubo_candidate_count": 10,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            results = run_stroke_shared_validation(
                seed=42,
                output_dir=tmp_dir,
                results_dir=tmp_dir,
                fixed_qubo=True,
                overwrite=True,
            )

            # Check that 5 heads were evaluated
            self.assertEqual(len(results), 5)
            methods = [r["method"] for r in results]
            self.assertEqual(
                methods,
                [
                    "Real LS",
                    "Binarized Ridge LS (refit scale+bias)",
                    "Direct Binary",
                    "STE Binary",
                    "QUBO Binary",
                ],
            )

            # Check that representation was shared across all 5 heads
            for r in results:
                self.assertEqual(r["representation_path"], "fake_rep.pt")
                self.assertEqual(r["representation_sha256"], "fake_hash_123")
                self.assertEqual(r["experiment_type"], "shared_backbone_head_comparison")
                self.assertEqual(r["dataset"], "stroke")
                self.assertEqual(r["backbone_parameter_count"], 1089)
                self.assertGreater(r["head_parameter_count"], 0)
                self.assertIn("head_fit_time_sec", r)

            # Check that canonical CSV was written
            canonical_csv = Path(tmp_dir) / "stroke_shared_results.csv"
            self.assertTrue(canonical_csv.exists())

    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.prepare_stroke_data")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.load_stroke_metadata")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.load_or_train_stroke_representation")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.train_real_ls")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.train_binarized_ls")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.train_direct_binary")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.train_ste_binary")
    @mock.patch("experiments.stroke_experiments.run_stroke_shared_validation.tune_from_representations")
    def test_shared_validation_with_qubo_tuning_pipeline(
        self,
        mock_tune,
        mock_ste,
        mock_direct,
        mock_bin_ridge,
        mock_real_ls,
        mock_load_rep,
        mock_meta,
        mock_prep,
    ):
        mock_meta.return_value = self.mock_metadata
        mock_load_rep.return_value = (self.mock_rep, Path("fake_rep.pt"), "reused")

        mock_real_ls.return_value = {"balanced_accuracy": 0.70, "f1": 0.20, "roc_auc": 0.75, "pr_auc": 0.15, "threshold": 0.5}
        mock_bin_ridge.return_value = {"balanced_accuracy": 0.71, "f1": 0.21, "roc_auc": 0.76, "pr_auc": 0.16, "threshold": 0.5}
        mock_direct.return_value = {"balanced_accuracy": 0.69, "f1": 0.19, "roc_auc": 0.74, "pr_auc": 0.14, "threshold": 0.0}
        mock_ste.return_value = {"balanced_accuracy": 0.72, "f1": 0.22, "roc_auc": 0.77, "pr_auc": 0.17, "threshold": 0.5, "initialization": "class_mean"}

        mock_winner = {
            "num_reads": 25,
            "num_sweeps": 250,
            "scale_multiplier": 1.0,
            "solver_seed": 1001,
            "validation_selection_score": 0.78,
            "validation_balanced_accuracy": 0.78,
            "validation_f1": 0.25,
            "validation_pr_auc": 0.20,
            "validation_roc_auc": 0.80,
            "runtime_sec": 1.2,
            "energy": -10.0,
            "candidate_source": "annealing",
            "candidate_count": 8,
            "base_alpha": 0.05,
            "_weights": torch.ones(64),
            "_alpha": 0.05,
            "_bias": -0.1,
            "_threshold": 0.2,
        }
        mock_tune.return_value = (mock_winner, [mock_winner])

        with tempfile.TemporaryDirectory() as tmp_dir:
            results = run_stroke_shared_validation(
                seed=42,
                output_dir=tmp_dir,
                results_dir=tmp_dir,
                fixed_qubo=False,
                overwrite=True,
            )

            qubo_row = [r for r in results if r["method"] == "QUBO Binary"][0]
            self.assertEqual(qubo_row["qubo_num_reads"], 25)
            self.assertEqual(qubo_row["qubo_num_sweeps"], 250)
            self.assertEqual(qubo_row["alpha_multiplier"], 1.0)
            self.assertEqual(qubo_row["qubo_grid_protocol"], "fixed_27_configuration_v1")
            self.assertEqual(qubo_row["threshold"], 0.2)


if __name__ == "__main__":
    unittest.main()

