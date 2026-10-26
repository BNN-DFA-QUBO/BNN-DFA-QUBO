"""Independent synthetic audit of the MNIST Binarized Ridge LS head."""
import inspect
import unittest

import torch
import torch.nn.functional as F

from experiments.MNIST_experiments.compare_ls_binary_qubo_shared import (
    binary_scores,
    fit_binarized_ridge,
)
from models.ls_heads import fit_ridge_classifier


def reference_binarized_fit(H_train, y_train, ridge_weights, num_classes):
    """Direct class-by-class reference for the documented train-only LS refit."""
    features = ridge_weights[:-1, :].detach().cpu().float()
    binary = torch.where(features >= 0, 1.0, -1.0)
    target_matrix = F.one_hot(y_train.cpu().long(), num_classes).float()
    scales, biases = [], []
    for cls in range(num_classes):
        z = H_train.cpu().float() @ binary[:, cls]
        target = target_matrix[:, cls]
        z_bar, target_bar = z.mean(), target.mean()
        zc, yc = z - z_bar, target - target_bar
        denominator = zc @ zc
        raw_scale = (zc @ yc) / denominator if denominator.item() > 1e-12 else torch.tensor(1.0)
        bias = target_bar - raw_scale * z_bar
        if raw_scale.item() < 0:
            binary[:, cls] *= -1
            raw_scale = -raw_scale
        scales.append(raw_scale)
        biases.append(bias)
    return binary, torch.stack(scales), torch.stack(biases)


class BinarizedRidgeAuditTests(unittest.TestCase):
    def test_shapes_class_order_and_bipolar_zero_convention(self):
        H = torch.tensor([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
        labels = torch.tensor([0, 0, 1, 1])
        ridge = torch.tensor([[-1., 1.], [0., 0.], [17., -23.]])
        weights, scales, biases = fit_binarized_ridge(H, labels, ridge, num_classes=2)
        self.assertEqual(tuple(weights.shape), (2, 2))
        self.assertEqual(tuple(scales.shape), (2,))
        self.assertEqual(tuple(biases.shape), (2,))
        self.assertEqual(weights.tolist(), [[-1.0, 1.0], [1.0, 1.0]])
        self.assertTrue(set(torch.unique(weights).tolist()) <= {-1.0, 1.0})
        scores = binary_scores(H, weights, scales, biases)
        self.assertEqual(tuple(scores.shape), (len(labels), 2))
        self.assertTrue(torch.equal(scores.argmax(1), labels))

    def test_refit_matches_independent_training_only_reference_and_minimizes_sse(self):
        torch.manual_seed(31)
        H_train = torch.randn(24, 5)
        y_train = torch.arange(24) % 3
        ridge = torch.randn(6, 3)
        # Force exact zeros so the bipolar convention is exercised, including class ordering.
        ridge[1, 0] = 0.0
        ridge[3, 2] = 0.0
        actual = fit_binarized_ridge(H_train, y_train, ridge, num_classes=3)
        expected = reference_binarized_fit(H_train, y_train, ridge, num_classes=3)
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor)

        weights, scales, biases = actual
        targets = F.one_hot(y_train, num_classes=3).float()
        fitted_scores = H_train @ (weights * scales.unsqueeze(0)) + biases
        fitted_sse = ((fitted_scores - targets) ** 2).sum().item()
        class_fit_sse_total = 0.0
        for cls in range(3):
            z = H_train @ weights[:, cls]
            class_fit_sse = ((scales[cls] * z + biases[cls] - targets[:, cls]) ** 2).sum().item()
            class_fit_sse_total += class_fit_sse
            for candidate_scale in (-3.0, -1.0, 0.0, 1.0, 3.0):
                candidate_bias = (targets[:, cls] - candidate_scale * z).mean()
                candidate_sse = ((candidate_scale * z + candidate_bias - targets[:, cls]) ** 2).sum().item()
                self.assertLessEqual(class_fit_sse, candidate_sse + 1e-4)
        self.assertAlmostEqual(class_fit_sse_total, fitted_sse, places=4)

    def test_fit_is_independent_of_holdout_arrays_and_scores_match_direct_formula(self):
        H_train = torch.tensor([[-3., 1.], [-1., 0.], [2., -1.], [4., 2.]])
        y_train = torch.tensor([0, 0, 1, 1])
        ridge = torch.tensor([[-1., 0.5], [0., 0.], [8., -9.]])
        first = fit_binarized_ridge(H_train, y_train, ridge, num_classes=2)
        # The fit API has no validation/test inputs, so those rows cannot affect refitting.
        parameters = inspect.signature(fit_binarized_ridge).parameters
        self.assertFalse({"H_val", "y_val", "H_test", "y_test"} & set(parameters))
        second = fit_binarized_ridge(H_train, y_train, ridge, num_classes=2)
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right)

        weights, scales, biases = first
        H_eval = torch.tensor([[-4., 3.], [0., 0.], [5., -2.]])
        scores = binary_scores(H_eval, weights, scales, biases)
        direct = torch.stack([
            torch.stack([
                sum(float(H_eval[n, j] * weights[j, c] * scales[c]) for j in range(2)) + biases[c]
                for c in range(2)
            ])
            for n in range(len(H_eval))
        ])
        torch.testing.assert_close(scores, direct)

    def test_ridge_coefficients_have_features_plus_intercept_by_class_shape(self):
        H = torch.tensor([[-2., 0.], [-1., 1.], [1., 0.], [2., -1.]])
        labels = torch.tensor([0, 0, 1, 1])
        coefficients = fit_ridge_classifier(H, labels, num_classes=2)
        self.assertEqual(tuple(coefficients.shape), (H.shape[1] + 1, 2))
        weights, _, _ = fit_binarized_ridge(H, labels, coefficients, num_classes=2)
        self.assertEqual(tuple(weights.shape), (H.shape[1], 2))


if __name__ == "__main__":
    unittest.main()
