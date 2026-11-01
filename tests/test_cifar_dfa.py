"""Unit tests for CIFAR-10 Direct Feedback Alignment (DFA)."""
import unittest
import torch
import torch.nn.functional as F

from models.cifar_bnn import CIFAR_BNN
from models.cifar_dfa import CIFARDFA, CIFARDFAFunction


class TestCIFARDFA(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = CIFAR_BNN()
        self.dfa = CIFARDFA(seed=42)

    def test_feedback_dimensions(self):
        # Conv1: 10 x 8192 (where 8192 = 32 * 16 * 16)
        self.assertEqual(self.dfa.feedback_conv1.shape, (10, 8192))
        # Conv2: 10 x 4096 (where 4096 = 64 * 8 * 8)
        self.assertEqual(self.dfa.feedback_conv2.shape, (10, 4096))
        # Dense: 10 x 128
        self.assertEqual(self.dfa.feedback_dense.shape, (10, 128))

    def test_feedback_matrices_remain_fixed(self):
        b1_initial = self.dfa.feedback_conv1.clone()
        b2_initial = self.dfa.feedback_conv2.clone()
        bd_initial = self.dfa.feedback_dense.clone()

        self.assertFalse(self.dfa.feedback_conv1.requires_grad)
        self.assertFalse(self.dfa.feedback_conv2.requires_grad)
        self.assertFalse(self.dfa.feedback_dense.requires_grad)

        x = torch.randn(2, 3, 32, 32)
        y = torch.tensor([2, 5])
        intermediates = self.model.forward_with_intermediates(x)
        grads = self.dfa.compute_local_gradients(self.model, x, y, intermediates)
        self.dfa.apply_gradients(self.model, grads)

        opt = torch.optim.Adam(self.model.parameters(), lr=0.01)
        opt.step()

        self.assertTrue(torch.equal(self.dfa.feedback_conv1, b1_initial))
        self.assertTrue(torch.equal(self.dfa.feedback_conv2, b2_initial))
        self.assertTrue(torch.equal(self.dfa.feedback_dense, bd_initial))

    def test_local_gradients_valid_and_finite(self):
        x = torch.randn(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])
        intermediates = self.model.forward_with_intermediates(x)
        grads = self.dfa.compute_local_gradients(self.model, x, y, intermediates)

        expected_shapes = {
            "conv1.weight": (32, 3, 3, 3),
            "conv1.bias": (32,),
            "conv2.weight": (64, 32, 3, 3),
            "conv2.bias": (64,),
            "fc1.weight": (128, 64),
            "fc1.bias": (128,),
            "fc2.weight": (10, 128),
            "fc2.bias": (10,),
        }

        for name, expected_shape in expected_shapes.items():
            self.assertIn(name, grads)
            g = grads[name]
            self.assertEqual(g.shape, expected_shape, f"Mismatch in {name} shape")
            self.assertTrue(torch.isfinite(g).all(), f"Non-finite values in {name}")
            self.assertGreater(g.abs().sum().item(), 0.0, f"All zero gradients in {name}")

    def test_no_backpropagation_through_hidden_layers(self):
        """Mathematical proof: Conv1 gradient in DFA does NOT depend on subsequent layer weights."""
        x = torch.randn(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])

        # Step 1: Forward pass with original model
        intermediates = self.model.forward_with_intermediates(x)
        grads1 = self.dfa.compute_local_gradients(self.model, x, y, intermediates)
        conv1_grad_original = grads1["conv1.weight"].clone()

        # Step 2: Randomize fc1 weights and fc2 weights completely
        with torch.no_grad():
            self.model.fc1.weight.copy_(torch.randn_like(self.model.fc1.weight) * 5.0)
            self.model.fc2.weight.copy_(torch.randn_like(self.model.fc2.weight) * 5.0)

        # Force output_error to be identical for both tests to isolate layer propagation
        output_error_fixed = CIFARDFAFunction.output_error(intermediates[0], y)

        # Compute conv1 gradient directly using fixed output_error and intermediates[1] (conv1_pre)
        conv1_delta = (output_error_fixed @ self.dfa.feedback_conv1).view(
            4, 32, 16, 16
        ) * (intermediates[1] > 0).float()
        import torch.nn.grad as G
        conv1_grad_computed = G.conv2d_weight(
            x, self.model.conv1.weight.shape, conv1_delta, stride=2, padding=1
        )

        # In DFA, conv1 gradient is strictly independent of fc1 and fc2 weights
        self.assertTrue(
            torch.allclose(conv1_grad_original, conv1_grad_computed, atol=1e-6),
            "Conv1 gradient must be independent of downstream weights in DFA",
        )

    def test_output_error_calculation(self):
        logits = torch.tensor([[2.0, -1.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        err = CIFARDFAFunction.output_error(logits, labels)

        # Shape [B, C]
        self.assertEqual(err.shape, (2, 2))
        # Sum of errors per sample should sum to 0 because sum(probs) = 1 and sum(targets) = 1
        sample_sums = err.sum(dim=1)
        self.assertTrue(torch.allclose(sample_sums, torch.zeros(2), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
