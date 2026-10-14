"""Fast synthetic tests for the shared Ridge-LS solver and Stroke head variants.

These tests do not download data or train a BNN/DFA representation.
Run with:
    python -m unittest experiments.stroke_experiments.test_stroke_ls_heads
"""
import unittest

import numpy as np
import torch

from models.ls_heads import fit_ridge_regression
from experiments.stroke_experiments.validate_stroke_ls_heads import (
    fit_legacy_ls,
    fit_legacy_binarized_ls,
    fit_ridge_ls,
    fit_ridge_binarized_ls,
    find_best_threshold_exact,
    score_continuous,
)


class RidgeRegressionTests(unittest.TestCase):
    def test_unregularized_scalar_fit_recovers_linear_target(self):
        torch.manual_seed(10)
        hidden = torch.randn(40, 5, dtype=torch.float64)
        expected_weights = torch.tensor([1.5, -0.75, 0.3, 0.0, 0.4], dtype=torch.float64)
        target = hidden @ expected_weights + 0.2
        weights = fit_ridge_regression(hidden, target, lambda_relative=0.0)

        self.assertEqual(tuple(weights.shape), (6, 1))
        design = torch.cat((hidden, torch.ones((len(hidden), 1), dtype=torch.float64)), dim=1)
        prediction = (design @ weights).squeeze(1)
        torch.testing.assert_close(prediction, target, rtol=1e-8, atol=1e-8)

    def test_rank_deficient_features_remain_finite_with_ridge(self):
        base = torch.arange(30, dtype=torch.float64).unsqueeze(1)
        hidden = torch.cat((base, base, base * 0, torch.ones_like(base)), dim=1)
        target = (base[:, 0] > 13).to(torch.float64)
        weights = fit_ridge_regression(hidden, target, lambda_relative=1e-4)

        self.assertEqual(tuple(weights.shape), (5, 1))
        self.assertTrue(torch.isfinite(weights).all().item())

    def test_multi_output_targets_preserve_output_shape(self):
        torch.manual_seed(11)
        hidden = torch.randn(32, 4, dtype=torch.float64)
        targets = torch.stack((hidden[:, 0] > 0, hidden[:, 1] > 0), dim=1).to(torch.float64)
        weights = fit_ridge_regression(hidden, targets, lambda_relative=1e-4)

        self.assertEqual(tuple(weights.shape), (5, 2))
        self.assertTrue(torch.isfinite(weights).all().item())

    def test_negative_regularization_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_ridge_regression(torch.ones(4, 2), torch.ones(4), lambda_relative=-1e-4)


class StrokeHeadComparisonTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.hidden = rng.normal(size=(80, 6))
        latent = 1.3 * self.hidden[:, 0] - 0.9 * self.hidden[:, 1] + 0.2 * self.hidden[:, 2]
        self.labels = (latent > np.median(latent)).astype(np.float64)

    def test_exact_threshold_search_handles_extreme_score_ranges(self):
        y = np.array([0, 0, 1, 1], dtype=np.int64)
        scores = np.array([-1e9, -1.0, 1.0, 1e9], dtype=np.float64)
        threshold, validation_ba = find_best_threshold_exact(y, scores)
        self.assertEqual(validation_ba, 1.0)
        np.testing.assert_array_equal((scores >= threshold).astype(int), y)

    def test_all_head_fits_produce_finite_scores(self):
        legacy = fit_legacy_ls(self.hidden, self.labels)
        legacy_binary, legacy_bias = fit_legacy_binarized_ls(self.hidden, self.labels)
        ridge = fit_ridge_ls(self.hidden, self.labels, lambda_relative=1e-4)
        ridge_binary, scale, bias = fit_ridge_binarized_ls(self.hidden, self.labels, ridge)

        continuous_scores = score_continuous(self.hidden, legacy)
        ridge_scores = score_continuous(self.hidden, ridge)
        legacy_binary_scores = self.hidden @ legacy_binary + legacy_bias
        ridge_binary_scores = (self.hidden @ ridge_binary) * scale + bias

        for values in (continuous_scores, ridge_scores, legacy_binary_scores, ridge_binary_scores):
            self.assertTrue(np.isfinite(values).all())
            self.assertEqual(values.shape, (len(self.labels),))

        self.assertEqual(legacy_binary.shape, (self.hidden.shape[1],))
        self.assertEqual(ridge_binary.shape, (self.hidden.shape[1],))
        self.assertTrue(set(np.unique(legacy_binary)).issubset({-1.0, 1.0}))
        self.assertTrue(set(np.unique(ridge_binary)).issubset({-1.0, 1.0}))
        self.assertGreaterEqual(scale, 0.0)


if __name__ == "__main__":
    unittest.main()
