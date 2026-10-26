"""Fast focused tests for Stroke QUBO solver validation and reporting."""
import csv
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from experiments.stroke_experiments import run_stroke_controlled
from experiments.stroke_experiments import stroke_dfa_qubo as qubo


class FakeResponse:
    def __init__(self, variables, samples, energies=None):
        self.variables = list(variables)
        self.record = SimpleNamespace(
            sample=np.asarray(samples),
            energy=np.asarray(energies if energies is not None else np.zeros(len(samples))),
        )
        self._samples = [
            {variable: row[index] for index, variable in enumerate(self.variables)}
            for row in np.asarray(samples)
        ]
        self.first = SimpleNamespace(sample=self._samples[0] if self._samples else {})

    def __len__(self):
        return len(self._samples)


class StrokeQuboValidationTests(unittest.TestCase):
    def setUp(self):
        self.H = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        self.target = torch.tensor([-1.0, 1.0, 1.0])

    def _optimize_with_response(self, response):
        with mock.patch.object(qubo.neal, "SimulatedAnnealingSampler") as sampler:
            sampler.return_value.sample_ising.return_value = response
            return qubo.optimize_binary_class(
                self.H, self.target, alpha=0.25, num_reads=5, seed=17, num_sweeps=33,
            )

    def test_optimizer_rejects_empty_response(self):
        response = FakeResponse([0, 1], [], [])
        with self.assertRaisesRegex(RuntimeError, "empty solver response"):
            self._optimize_with_response(response)

    def test_optimizer_rejects_missing_variable(self):
        response = FakeResponse([0], [[1]], [0.0])
        with self.assertRaisesRegex(RuntimeError, "missing expected variables: \\[1\\]"):
            self._optimize_with_response(response)

    def test_optimizer_rejects_invalid_spin_in_any_returned_sample(self):
        response = FakeResponse([0, 1], [[1, -1], [-1, 0]], [0.0, 1.0])
        with self.assertRaisesRegex(RuntimeError, "invalid spin .*0.*sample 1"):
            self._optimize_with_response(response)

    def test_optimizer_forwards_reads_sweeps_and_seed(self):
        response = FakeResponse([0, 1], [[1, -1]], [0.0])
        with mock.patch.object(qubo.neal, "SimulatedAnnealingSampler") as sampler:
            sampler.return_value.sample_ising.return_value = response
            result = qubo.optimize_binary_class(
                self.H, self.target, alpha=0.25, num_reads=5, seed=17, num_sweeps=33,
            )
        self.assertEqual(result.tolist(), [1.0, -1.0])
        sampler.return_value.sample_ising.assert_called_once()
        self.assertEqual(sampler.return_value.sample_ising.call_args.kwargs,
                         {"num_reads": 5, "num_sweeps": 33, "seed": 17})

    def test_optimizer_rejects_invalid_read_sweep_and_seed_types(self):
        for name, values in (
            ("num_reads", (0, -1, True, 2.5, "3")),
            ("num_sweeps", (0, -1, False, 2.5, "3")),
            ("seed", (-1, 2 ** 32, True, 2.5, "3")),
        ):
            for value in values:
                with self.subTest(name=name, value=value):
                    kwargs = {"num_reads": 5, "num_sweeps": 5, "seed": 17}
                    kwargs[name] = value
                    with mock.patch.object(qubo.neal, "SimulatedAnnealingSampler"):
                        with self.assertRaisesRegex(ValueError, name):
                            qubo.optimize_binary_class(
                                self.H, self.target, alpha=0.25, **kwargs,
                            )

    def test_optimizer_rejects_nonfinite_or_nonpositive_alpha(self):
        for alpha in (float("nan"), float("inf"), -float("inf"), 0.0):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    qubo.optimize_binary_class(
                        self.H, self.target, alpha=alpha, num_reads=5, seed=17,
                    )

    def test_candidate_search_validates_all_samples_and_its_arguments(self):
        with mock.patch.object(qubo, "fit_ridge_regression",
                               return_value=torch.zeros(3, 1)), \
             mock.patch.object(qubo.neal, "SimulatedAnnealingSampler") as sampler:
            sampler.return_value.sample_ising.return_value = FakeResponse(
                [0, 1], [[1, -1], [-1, 0]], [0.0, 1.0]
            )
            with self.assertRaisesRegex(RuntimeError, "invalid spin .*0.*sample 1"):
                qubo.sample_binary_candidates(
                    self.H, self.target, 0.25, num_reads=2, seed=17,
                    alpha_multipliers=(1.0,), top_k=1, num_sweeps=3,
                    random_seed_starts=0,
                )

        for name, value in (("num_reads", True), ("num_sweeps", 1.5), ("seed", -1)):
            with self.subTest(name=name, value=value):
                kwargs = {"num_reads": 2, "num_sweeps": 3, "seed": 17}
                kwargs[name] = value
                with self.assertRaisesRegex(ValueError, name):
                    qubo.sample_binary_candidates(
                        self.H, self.target, 0.25, alpha_multipliers=(1.0,), **kwargs,
                    )
        for scale in (float("nan"), float("inf"), 0.0):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "base_alpha"):
                    qubo.sample_binary_candidates(
                        self.H, self.target, scale, num_reads=2, seed=17,
                        alpha_multipliers=(1.0,),
                    )

    def test_ising_energy_matches_independent_objective_up_to_constant(self):
        H = torch.tensor([
            [0.2, 1.3, -0.4],
            [1.1, -0.2, 0.5],
            [-0.7, 0.8, 1.6],
            [1.3, 0.4, -0.1],
        ], dtype=torch.float64)
        target = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
        alpha = 0.7
        h, J = qubo.build_ising_model(H, target, alpha)
        Hc = H - H.mean(0, keepdim=True)
        yc = target - target.mean()
        differences = []
        for spins in itertools.product((-1.0, 1.0), repeat=H.shape[1]):
            weights = torch.tensor(spins, dtype=torch.float64)
            direct = ((alpha * (Hc @ weights) - yc) ** 2).sum().item()
            ising = qubo._ising_energy(spins, h, J)
            differences.append(direct - ising)
        self.assertLess(max(differences) - min(differences), 1e-10)

    def test_train_qubo_ranks_validation_only_and_records_complete_search_timing(self):
        H_train = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        H_val = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        H_test = torch.tensor([[2.0, 0.0], [-2.0, 0.0]])
        y_train = torch.tensor([0.0, 1.0])
        y_val = torch.tensor([0.0, 1.0])
        # Unique sentinel values make any test-label leakage into ranking visible.
        y_test = torch.tensor([7.0, 8.0])
        data = {
            "H_train": H_train, "H_val": H_val, "H_test": H_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "training_seed": 42, "seed": 42,
        }
        candidates = [
            {"weights": torch.tensor([1.0, 1.0]), "alpha": 1.0,
             "alpha_multiplier": 0.5, "energy": 0.0, "candidate_source": "seed"},
            {"weights": torch.tensor([-1.0, 1.0]), "alpha": 1.0,
             "alpha_multiplier": 1.0, "energy": 0.0, "candidate_source": "annealing"},
        ]
        clock = [0.0]
        metric_labels = []
        threshold_labels = []

        def advance(amount):
            clock[0] += amount

        def fake_metrics(labels, scores, threshold):
            labels = np.asarray(labels)
            metric_labels.append(labels.copy())
            if np.array_equal(labels, y_test.numpy()):
                advance(100.0)  # Deliberately outside the search timer.
                return {"balanced_accuracy": 0.5, "precision": 0.0, "recall": 0.0,
                        "f1": 0.0, "roc_auc": 0.5, "pr_auc": 0.5,
                        "tn": 0, "fp": 0, "fn": 0, "tp": 0,
                        "predicted_positive": 0, "actual_positive": 0,
                        "threshold": threshold}
            advance(3.0)
            # The first candidate wins based only on the validation scores.
            quality = 0.9 if scores[0] > 0 else 0.1
            return {"balanced_accuracy": quality, "precision": quality, "recall": quality,
                    "f1": quality, "roc_auc": quality, "pr_auc": quality,
                    "tn": 0, "fp": 0, "fn": 0, "tp": 0,
                    "predicted_positive": 1, "actual_positive": 1,
                    "threshold": threshold}

        def fake_threshold(labels, scores):
            threshold_labels.append(np.asarray(labels).copy())
            advance(1.0)
            return 0.0, 0.5

        def fake_candidates(**kwargs):
            advance(2.0)  # Mock annealing, sample collection, and coordinate refinement.
            return candidates

        with mock.patch.object(qubo, "set_seed"), \
             mock.patch.object(qubo, "load_stroke_head_representation", return_value=data), \
             mock.patch.object(qubo, "standardize_hidden_features",
                               return_value=(H_train, H_val, H_test, {"transform": "test", "constant_units": 0})), \
             mock.patch.object(qubo, "calculate_alpha", return_value=1.0), \
             mock.patch.object(qubo, "sample_binary_candidates", side_effect=fake_candidates), \
             mock.patch.object(qubo, "recover_bias", return_value=0.0), \
             mock.patch.object(qubo, "find_best_threshold", side_effect=fake_threshold), \
             mock.patch.object(qubo, "calculate_metrics", side_effect=fake_metrics), \
             mock.patch.object(qubo.time, "perf_counter", side_effect=lambda: clock[0]):
            result = qubo.train_qubo("synthetic.pt", seed=42)

        self.assertEqual(result["alpha_multiplier"], 0.5)
        self.assertEqual(result["qubo_candidate_generation_seconds"], 2.0)
        self.assertEqual(result["qubo_validation_selection_seconds"], 8.0)
        self.assertEqual(result["qubo_search_seconds"], 10.0)
        self.assertEqual(len(threshold_labels), 2)
        self.assertTrue(all(np.array_equal(labels, y_val.numpy()) for labels in threshold_labels))
        self.assertEqual(len(metric_labels), 3)
        self.assertTrue(all(np.array_equal(labels, y_val.numpy()) for labels in metric_labels[:2]))
        self.assertTrue(np.array_equal(metric_labels[-1], y_test.numpy()))

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            path = Path(tmp) / "controlled.csv"
            row = {"method": "QUBO Binary", **result}
            run_stroke_controlled.save_results([row], path)
            with path.open(newline="", encoding="utf-8") as handle:
                record = next(iter(csv.DictReader(handle)))
            self.assertEqual(float(record["qubo_search_seconds"]), 10.0)
            self.assertEqual(float(record["qubo_candidate_generation_seconds"]), 2.0)
            self.assertEqual(float(record["qubo_validation_selection_seconds"]), 8.0)


if __name__ == "__main__":
    unittest.main()
