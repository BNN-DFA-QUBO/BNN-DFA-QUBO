"""Fast synthetic regression tests for model-math invariants.

Run: python -m unittest tests.test_model_invariants
These tests require installed project dependencies but do not download datasets.
"""
import itertools
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from models.dfa import DFAFunction
from models.qubo_head import build_ising_model
from models.stroke_dfa import StrokeDFAFunction
from experiments.MNIST_experiments.dfa_binary_head_ste import binary_ste
from utils.stroke_metrics import find_best_threshold


def _ising_energy(spins, h, J):
    return sum(float(h.get(i, 0.0)) * spins[i] for i in range(len(spins))) + sum(
        float(value) * spins[i] * spins[j] for (i, j), value in J.items()
    )


class QUBOEnergyTests(unittest.TestCase):
    def _assert_objective_equivalence(self, H, target, alpha):
        h, J = build_ising_model(H, target, alpha)
        H = H.float()
        target = target.float()
        Hc = H - H.mean(dim=0, keepdim=True)
        yc = target - target.mean()
        offsets = []
        for spins in itertools.product((-1.0, 1.0), repeat=H.shape[1]):
            w = torch.tensor(spins, dtype=H.dtype)
            direct = ((alpha * (Hc @ w) - yc) ** 2).sum().item()
            offsets.append(direct - _ising_energy(spins, h, J))
        # The diagonal quadratic terms and target norm are constants for spins.
        self.assertLess(max(offsets) - min(offsets), 1e-4)

    def test_ising_energy_matches_direct_objective_exhaustively(self):
        torch.manual_seed(19)
        H = torch.randn(7, 4)
        target = torch.tensor([-1., 1., -1., 1., 1., -1., 1.])
        self._assert_objective_equivalence(H, target, 0.37)

    def test_ising_energy_handles_rank_deficient_features(self):
        H = torch.tensor([[1., 1., 0.], [2., 2., 0.], [3., 3., 0.], [4., 4., 0.]])
        target = torch.tensor([-1., -1., 1., 1.])
        self._assert_objective_equivalence(H, target, 0.2)

    def test_pairwise_cutoff_has_bounded_energy_error_around_threshold(self):
        from models.qubo_head import build_ising_model

        target = torch.zeros(4)
        base = torch.tensor([-1., 1., -1., 1.])
        maximum_discrepancy = 0.0
        for desired_coupling, should_be_retained in ((0.9e-12, False), (1.1e-12, True)):
            ratio = desired_coupling / 8.0
            H = torch.stack((base, base * ratio), dim=1)
            h, J = build_ising_model(H, target, alpha=1.0)
            if should_be_retained:
                self.assertIn((0, 1), J)
            else:
                self.assertNotIn((0, 1), J)

            # Independently calculate the full centered least-squares energy in
            # float64, then compare its spin-dependent variation with h/J.
            H64 = H.double()
            target64 = target.double()
            Hc = H64 - H64.mean(dim=0, keepdim=True)
            yc = target64 - target64.mean()
            offsets = []
            for spins in itertools.product((-1.0, 1.0), repeat=2):
                w = torch.tensor(spins, dtype=torch.float64)
                direct = ((Hc @ w - yc) ** 2).sum().item()
                approximate = _ising_energy(spins, h, J)
                offsets.append(direct - approximate)
            discrepancy = max(offsets) - min(offsets)
            maximum_discrepancy = max(maximum_discrepancy, discrepancy)

        # One omitted 0.9e-12 coupler can vary by at most 2|J| over spins;
        # this fixture measures 1.8e-12 (absolute energy units).
        self.assertAlmostEqual(maximum_discrepancy, 1.8e-12, delta=2e-15)
        self.assertLessEqual(maximum_discrepancy, 2e-12)


class GradientScalingTests(unittest.TestCase):
    def test_mnist_dfa_error_matches_mean_cross_entropy_gradient(self):
        logits = torch.tensor([[1.2, -0.4, 0.2], [-0.3, 0.8, 0.1]], requires_grad=True)
        labels = torch.tensor([0, 2])
        expected = torch.autograd.grad(F.cross_entropy(logits, labels), logits)[0]
        torch.testing.assert_close(DFAFunction.output_error(logits, labels), expected)

    def test_stroke_error_requires_one_batch_division_for_mean_bce(self):
        logits = torch.tensor([[0.7], [-1.1], [0.2]], requires_grad=True)
        labels = torch.tensor([[1.], [0.], [1.]])
        expected = torch.autograd.grad(
            F.binary_cross_entropy_with_logits(logits, labels, reduction="mean"), logits
        )[0]
        actual = StrokeDFAFunction.output_error(logits, labels) / len(labels)
        torch.testing.assert_close(actual, expected)


class STEAndThresholdTests(unittest.TestCase):
    def test_ste_has_hard_binary_forward_and_identity_gradient(self):
        weights = torch.tensor([-0.4, 0.0, 0.8], requires_grad=True)
        binary = binary_ste(weights)
        self.assertEqual(binary.tolist(), [-1.0, 1.0, 1.0])
        binary.sum().backward()
        torch.testing.assert_close(weights.grad, torch.ones_like(weights))

    def test_threshold_search_matches_brute_force_unique_boundaries(self):
        y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        scores = np.array([-2., -1., -1., 0.5, 0.5, 2., 3., 3.])
        threshold, ba = find_best_threshold(y, scores)
        candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
        brute = [(balanced_accuracy_score(y, (scores >= t).astype(int)), float(t)) for t in candidates]
        best_ba = max(score for score, _ in brute)
        best_threshold = min(t for score, t in brute if score == best_ba)
        self.assertAlmostEqual(ba, best_ba)
        self.assertEqual(threshold, best_threshold)


if __name__ == "__main__":
    unittest.main()
