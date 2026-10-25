"""Cheap mathematical regression tests for the MNIST/Stroke/QUBO heads.

Run with: python -m unittest tests.test_model_invariants
No dataset download or model training is required.
"""
import itertools
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from models.dfa import DFAFunction
from models.qubo_head import build_ising_model
from models.stroke_dfa import StrokeDFAFunction
from experiments.MNIST_experiments.dfa_binary_head_ste import binary_ste
from utils.stroke_metrics import find_best_threshold


class QUBOEnergyTests(unittest.TestCase):
    @staticmethod
    def energy(spins, h, J):
        return sum(h.get(i, 0.0) * spins[i] for i in range(len(spins))) + sum(
            value * spins[i] * spins[j] for (i, j), value in J.items()
        )

    def test_ising_energy_matches_centered_residual_objective_up_to_constant(self):
        torch.manual_seed(19)
        H = torch.randn(7, 4)
        target = torch.tensor([-1., 1., -1., 1., 1., -1., 1.])
        alpha = 0.37
        h, J = build_ising_model(H, target, alpha)
        Hc = H.float() - H.float().mean(dim=0, keepdim=True)
        yc = target.float() - target.float().mean()
        rows = []
        for bits in itertools.product((-1., 1.), repeat=H.shape[1]):
            w = torch.tensor(bits)
            residual = torch.sum((alpha * (Hc @ w) - yc) ** 2).item()
            rows.append((self.energy(bits, h, J), residual))
        # Diagonal quadratic terms are constant for spin variables in {-1,+1}.
        offsets = [residual - energy for energy, residual in rows]
        self.assertLess(max(offsets) - min(offsets), 1e-4)

    def test_ising_energy_equivalence_on_rank_deficient_features(self):
        H = torch.tensor([[1., 1., 0.], [2., 2., 0.], [3., 3., 0.], [4., 4., 0.]])
        target = torch.tensor([-1., -1., 1., 1.])
        h, J = build_ising_model(H, target, 0.2)
        values = []
        for bits in itertools.product((-1., 1.), repeat=H.shape[1]):
            w = torch.tensor(bits)
            Hc = H - H.mean(0, keepdim=True)
            yc = target - target.mean()
            objective = ((0.2 * Hc @ w - yc) ** 2).sum().item()
            values.append(objective - self.energy(bits, h, J))
        self.assertLess(max(values) - min(values), 1e-4)


class GradientScalingTests(unittest.TestCase):
    def test_mnist_output_error_is_mean_cross_entropy_logit_gradient(self):
        logits = torch.tensor([[1.2, -0.4, 0.2], [-0.3, 0.8, 0.1]], requires_grad=True)
        labels = torch.tensor([0, 2])
        expected = torch.autograd.grad(F.cross_entropy(logits, labels), logits)[0]
        actual = DFAFunction.output_error(logits, labels)
        torch.testing.assert_close(actual, expected)

    def test_stroke_output_error_is_sum_gradient_and_batch_division_gives_mean(self):
        logits = torch.tensor([[0.7], [-1.1], [0.2]], requires_grad=True)
        labels = torch.tensor([[1.], [0.], [1.]])
        mean_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        expected_mean = torch.autograd.grad(mean_loss, logits)[0]
        actual_mean = StrokeDFAFunction.output_error(logits, labels) / labels.shape[0]
        torch.testing.assert_close(actual_mean, expected_mean)


class STEAndThresholdTests(unittest.TestCase):
    def test_ste_forward_is_hard_binary_and_backward_is_identity(self):
        weights = torch.tensor([-0.4, 0.0, 0.8], requires_grad=True)
        result = binary_ste(weights)
        self.assertEqual(result.tolist(), [-1.0, 1.0, 1.0])
        result.sum().backward()
        torch.testing.assert_close(weights.grad, torch.ones_like(weights))

    def test_threshold_search_matches_brute_force_unique_boundaries(self):
        y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        scores = np.array([-2.0, -1.0, -1.0, 0.5, 0.5, 2.0, 3.0, 3.0])
        threshold, score = find_best_threshold(y, scores)
        candidates = np.r_[np.unique(scores), np.nextafter(np.max(scores), np.inf)]
        brute = []
        for candidate in candidates:
            pred = (scores >= candidate).astype(int)
            sensitivity = (pred[y == 1] == 1).mean()
            specificity = (pred[y == 0] == 0).mean()
            brute.append((float((sensitivity + specificity) / 2), float(candidate)))
        best_score = max(item[0] for item in brute)
        self.assertAlmostEqual(score, best_score)
        # Tie behavior is part of the threshold API: choose the first/lower boundary.
        best_threshold = min(item[1] for item in brute if item[0] == best_score)
        self.assertEqual(threshold, best_threshold)


if __name__ == "__main__":
    unittest.main()
