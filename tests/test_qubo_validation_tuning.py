"""Fast unit tests for the bounded MNIST QUBO tuning helpers."""
import argparse
import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch

from experiments.MNIST_experiments.tune_qubo_validation import (
    DEFAULT_NUM_READS,
    DEFAULT_NUM_SWEEPS,
    DEFAULT_SCALE_MULTIPLIERS,
    apply_scale_multiplier,
    _parse_grid,
    select_configuration,
    validate_grid,
    _accuracy,
)
import experiments.MNIST_experiments.tune_qubo_validation as tuning
from models.qubo_head import optimize_binary_class, optimize_binary_classifier


class QuboValidationTuningTests(unittest.TestCase):
    def test_default_grid_is_exactly_27_finite_configurations(self):
        grid = validate_grid(DEFAULT_NUM_READS, DEFAULT_NUM_SWEEPS, DEFAULT_SCALE_MULTIPLIERS)
        self.assertEqual(len(grid), 27)
        self.assertEqual(grid[0], (25, 250, 0.5))
        self.assertEqual(grid[-1], (100, 1000, 2.0))

    def test_grid_rejects_empty_nonpositive_duplicate_and_oversized_inputs(self):
        for reads, sweeps, multipliers in [
            ([], [250, 500, 1000], [0.5, 1.0, 2.0]),
            ([0, 50, 100], [250, 500, 1000], [0.5, 1.0, 2.0]),
            ([25, 50, 100], [0, 500, 1000], [0.5, 1.0, 2.0]),
            ([25, 50, 100], [250, 500, 1000], [0.5, 0.0, 2.0]),
            ([25, 50, 100], [250, 500, 1000], [0.5, 1.0, float("inf")]),
            ([25, 25, 100], [250, 500, 1000], [0.5, 1.0, 2.0]),
            ([25, 50, 100], [250, 250, 1000], [0.5, 1.0, 2.0]),
            ([50, 25, 100], [250, 500, 1000], [0.5, 1.0, 2.0]),
        ]:
            with self.subTest(reads=reads, sweeps=sweeps, multipliers=multipliers):
                with self.assertRaises(ValueError):
                    validate_grid(reads, sweeps, multipliers)

    def test_cli_grid_parser_rejects_malformed_and_nonpositive_values(self):
        for raw in ("", "1,,2", "abc", "0,2", "1,-2"):
            with self.subTest(raw=raw):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _parse_grid(raw, int, "grid")

    def test_scale_multiplier_is_applied_to_every_ridge_scale(self):
        base = torch.tensor([0.2, 0.5, 1.0])
        torch.testing.assert_close(apply_scale_multiplier(base, 0.5), torch.tensor([0.1, 0.25, 0.5]))

    def test_tie_break_order_is_reads_multiplier_distance_then_sweeps(self):
        rows = [
            {"num_reads": 50, "num_sweeps": 250, "scale_multiplier": 1.0, "validation_accuracy": 88.0},
            {"num_reads": 25, "num_sweeps": 1000, "scale_multiplier": 0.5, "validation_accuracy": 88.0},
            {"num_reads": 25, "num_sweeps": 250, "scale_multiplier": 1.0, "validation_accuracy": 88.0},
        ]
        self.assertIs(select_configuration(rows), rows[2])
        rows.append({"num_reads": 25, "num_sweeps": 250, "scale_multiplier": 0.5,
                     "validation_accuracy": 88.0})
        self.assertIs(select_configuration(rows), rows[2])
        rows.append({"num_reads": 25, "num_sweeps": 1000, "scale_multiplier": 1.0,
                     "validation_accuracy": 88.0})
        self.assertIs(select_configuration(rows), rows[2])

    def test_selection_uses_validation_accuracy_only(self):
        rows = [
            {"num_reads": 25, "num_sweeps": 250, "scale_multiplier": 1.0,
             "validation_accuracy": 80.0, "test_accuracy": 99.0},
            {"num_reads": 50, "num_sweeps": 500, "scale_multiplier": 1.0,
             "validation_accuracy": 81.0, "test_accuracy": 10.0},
        ]
        self.assertIs(select_configuration(rows), rows[1])

    def test_accuracy_helper_scores_only_the_supplied_split(self):
        scores = torch.tensor([[2., 0.], [0., 2.]])
        labels = torch.tensor([0, 0])
        self.assertEqual(_accuracy(scores, labels), 50.0)

    def test_num_sweeps_reaches_the_neal_sample_ising_api(self):
        response = mock.MagicMock()
        response.first.sample = {0: 1, 1: -1}
        with mock.patch("models.qubo_head.neal.SimulatedAnnealingSampler") as sampler_type:
            sampler_type.return_value.sample_ising.return_value = response
            result = optimize_binary_class(
                torch.tensor([[0., 1.], [1., 0.], [2., 1.]]),
                torch.tensor([-1., 1., 1.]), alpha=0.2,
                num_reads=25, seed=9, num_sweeps=777,
            )
        self.assertEqual(result.tolist(), [1.0, -1.0])
        sampler = sampler_type.return_value
        sampler.sample_ising.assert_called_once()
        self.assertEqual(sampler.sample_ising.call_args.kwargs["num_sweeps"], 777)

    def test_classifier_passes_num_sweeps_to_each_class_optimizer(self):
        with mock.patch("models.qubo_head.optimize_binary_class", return_value=torch.ones(2)) as optimizer:
            optimize_binary_classifier(torch.zeros(3, 2), torch.zeros(3, 2), torch.ones(2),
                                       num_classes=2, num_reads=50, seed=6, num_sweeps=321)
        self.assertEqual(optimizer.call_count, 2)
        self.assertEqual(optimizer.call_args.kwargs["num_sweeps"], 321)

    def test_test_accuracy_is_computed_after_selection_and_once(self):
        events = []
        class Dataset:
            targets = [0, 1] * 10
            def __len__(self): return 20
            def __getitem__(self, index): return torch.zeros(1), self.targets[index]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(1, 1)

        features = torch.zeros(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        artifact = {
            "H_train": features, "y_train": labels,
            "H_validation": features, "y_validation": labels,
            "H_test": features, "y_test": labels,
            "metadata": {"sample_counts": {"train": 4, "validation": 4, "test": 4},
                         "split_protocol": "synthetic"},
            "artifact_id": "mock-artifact",
        }
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp, \
             mock.patch.object(tuning, "set_seed", return_value=7), \
             mock.patch.object(tuning, "get_mnist_loaders", return_value=(
                 torch.utils.data.DataLoader(Dataset(), batch_size=4),
                 torch.utils.data.DataLoader(Dataset(), batch_size=4))), \
             mock.patch.object(tuning, "load_or_train_mnist_representation",
                               return_value=(artifact, Path(temp) / "artifact.pt", "reused")), \
             mock.patch.object(tuning, "create_binary_targets", return_value=torch.ones(4, 10)), \
             mock.patch.object(tuning, "calculate_ridge_ls_scales", return_value=torch.ones(10)), \
             mock.patch.object(tuning, "optimize_binary_classifier", return_value=torch.ones(2, 10)) as mocked_optimizer, \
             mock.patch.object(tuning, "calculate_final_scale_and_bias",
                               return_value=(torch.ones(2, 10), torch.ones(10), torch.zeros(10))), \
             mock.patch.object(tuning, "_accuracy", side_effect=lambda *a: events.append("score") or 50.0):
            # Use the real selection implementation while recording its position.
            real_select = tuning.select_configuration
            tuning.select_configuration = lambda rows: (events.append("select") or real_select(rows))
            try:
                summary, tuning_csv, summary_json = tuning.run_tuning(
                    seed=7, epochs=1, validation_fraction=0.2, output_dir=temp,
                )
                with tuning_csv.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 27)
                self.assertIn("runtime_sec", rows[0])
                self.assertIn("num_sweeps", rows[0])
                self.assertTrue(tuning_csv.name.startswith("qubo_validation_tuning_27config_"))
                self.assertEqual(summary["grid_protocol"], "fixed_27_configuration_v1")
                self.assertTrue(summary_json.exists())
            finally:
                tuning.select_configuration = real_select
        self.assertEqual(events.count("score"), 28)
        self.assertEqual(events[27], "select")
        self.assertEqual(mocked_optimizer.call_count, 27)
        self.assertEqual(mocked_optimizer.call_args_list[0].kwargs["num_sweeps"], 250)
        self.assertEqual(mocked_optimizer.call_args_list[-1].kwargs["num_sweeps"], 1000)


if __name__ == "__main__":
    unittest.main()
