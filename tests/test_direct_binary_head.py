"""Synthetic checks for the MNIST direct binary classifier head."""
import unittest

import torch

from experiments.MNIST_experiments.dfa_direct_binary_head import optimize_binary_head


class DirectBinaryHeadTests(unittest.TestCase):
    def test_dimensions_bipolar_scores_and_fixed_seed_determinism(self):
        features = torch.tensor([
            [-1.0, 0.2, 0.5], [-0.8, 0.1, 0.3],
            [0.9, -0.2, -0.4], [1.1, -0.1, -0.2],
            [0.1, 1.0, -0.5], [-0.2, 0.8, -0.7],
        ])
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        torch.manual_seed(23)
        first_weights, first_bias, _ = optimize_binary_head(
            features, labels, num_classes=3, steps=12, lr=0.02,
        )
        torch.manual_seed(23)
        second_weights, second_bias, _ = optimize_binary_head(
            features, labels, num_classes=3, steps=12, lr=0.02,
        )
        self.assertEqual(tuple(first_weights.shape), (3, 3))
        self.assertEqual(tuple(first_bias.shape), (3,))
        self.assertTrue(set(torch.unique(first_weights).tolist()) <= {-1.0, 1.0})
        scores = features @ first_weights + first_bias
        self.assertEqual(tuple(scores.shape), (len(labels), 3))
        self.assertTrue(torch.isfinite(scores).all())
        torch.testing.assert_close(first_weights, second_weights, rtol=0, atol=0)
        torch.testing.assert_close(first_bias, second_bias, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
