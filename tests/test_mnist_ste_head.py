"""Regression tests for the MNIST straight-through-estimator classifier head."""
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import torch

from experiments.MNIST_experiments.dfa_binary_head_ste import (
    binary_ste,
    train_binary_head_ste,
)


class TestSTEHead(unittest.TestCase):
    def test_binary_ste_has_binary_forward_and_identity_gradient(self):
        weights = torch.tensor([-0.25, 0.7, 0.0], requires_grad=True)

        binary = binary_ste(weights)

        self.assertEqual(binary.tolist(), [-1.0, 1.0, 1.0])
        binary.sum().backward()
        self.assertTrue(torch.equal(weights.grad, torch.ones_like(weights)))

    def test_training_runs_requested_number_of_optimizer_steps(self):
        torch.manual_seed(123)
        features = torch.randn(24, 8)
        labels = torch.arange(24, dtype=torch.long) % 3
        steps = []
        real_adam = torch.optim.Adam

        class CountingAdam:
            def __init__(self, params, lr):
                self.optimizer = real_adam(params, lr=lr)

            def zero_grad(self):
                return self.optimizer.zero_grad()

            def step(self):
                steps.append(1)
                return self.optimizer.step()

        with patch(
            "experiments.MNIST_experiments.dfa_binary_head_ste.torch.optim.Adam",
            side_effect=CountingAdam,
        ):
            with redirect_stdout(StringIO()):
                binary_weights, bias, accuracy = train_binary_head_ste(
                    features,
                    labels,
                    num_classes=3,
                    epochs=4,
                    lr=0.01,
                )

        self.assertEqual(len(steps), 4)
        self.assertEqual(tuple(binary_weights.shape), (8, 3))
        self.assertEqual(tuple(bias.shape), (3,))
        self.assertTrue(set(torch.unique(binary_weights).tolist()) <= {-1.0, 1.0})
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 100.0)


if __name__ == "__main__":
    unittest.main()
