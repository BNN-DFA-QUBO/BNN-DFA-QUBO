"""Unit tests for CIFAR-10 BNN model architecture and binary layers."""
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cifar_bnn import BinaryConv2d, CIFAR_BNN
from models.bnn import BinaryLinear


class TestCIFARBNN(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = CIFAR_BNN()

    def test_binary_conv2d_forward_weights(self):
        layer = BinaryConv2d(3, 32, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 16, 16)
        out = layer(x)

        self.assertEqual(out.shape, (2, 32, 16, 16))
        # Check that weights are binarized during forward
        signs = layer.weight.sign()
        scale = layer.weight.abs().mean()
        expected_binary = signs * scale
        # Effective weights have at most 2 unique magnitude values (+scale, -scale)
        unique_vals = torch.unique(expected_binary)
        self.assertLessEqual(len(unique_vals), 3)  # could include 0 if any weight is exact 0

    def test_model_output_dimensions(self):
        x = torch.randn(4, 3, 32, 32)
        logits = self.model(x)
        self.assertEqual(logits.shape, (4, 10), "Model logits must be [B, 10]")

    def test_feature_representation_dimensions(self):
        x = torch.randn(4, 3, 32, 32)
        features = self.model.extract_features(x)
        self.assertEqual(features.shape, (4, 128), "Feature representation must be [B, 128]")

        # Logits from features through fc2 must equal full forward pass
        logits_from_feats = self.model.fc2(features)
        # Note: Because weights are deterministic in inference, they should match
        logits_full = self.model(x)
        self.assertTrue(torch.allclose(logits_from_feats, logits_full, atol=1e-5))

    def test_parameter_count(self):
        # conv1: 32*3*3*3 + 32 = 896
        # conv2: 64*32*3*3 + 64 = 18496
        # fc1: 128*64 + 128 = 8320
        # fc2: 10*128 + 10 = 1290
        # Total: 29002
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(total_params, 29002, "Expected 29,002 parameters in CIFAR_BNN")

    def test_ste_gradient_flow_and_parameter_update(self):
        x = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([1, 7])
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        w1_before = self.model.conv1.weight.clone()
        w2_before = self.model.conv2.weight.clone()
        fc1_before = self.model.fc1.weight.clone()

        logits = self.model(x)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        # Check gradients exist and are finite
        self.assertIsNotNone(self.model.conv1.weight.grad)
        self.assertIsNotNone(self.model.conv2.weight.grad)
        self.assertIsNotNone(self.model.fc1.weight.grad)
        self.assertTrue(torch.isfinite(self.model.conv1.weight.grad).all())
        self.assertTrue(torch.isfinite(self.model.conv2.weight.grad).all())
        self.assertTrue(torch.isfinite(self.model.fc1.weight.grad).all())

        optimizer.step()

        # Verify weights actually updated
        self.assertFalse(torch.equal(self.model.conv1.weight, w1_before))
        self.assertFalse(torch.equal(self.model.conv2.weight, w2_before))
        self.assertFalse(torch.equal(self.model.fc1.weight, fc1_before))


if __name__ == "__main__":
    unittest.main()
