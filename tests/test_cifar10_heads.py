"""Unit tests for all five CIFAR-10 classifier heads on synthetic representations."""
import unittest
import torch

from experiments.CIFAR10_experiments.cifar_dfa_binarized_ls import (
    evaluate_binarized_head,
    predict_binary_scores,
)
from experiments.CIFAR10_experiments.cifar_dfa_direct_binary import (
    evaluate_direct_binary_head,
    predict_direct_binary,
)
from experiments.CIFAR10_experiments.cifar_dfa_least_squares import (
    fit_and_evaluate_ridge,
)
from experiments.CIFAR10_experiments.cifar_dfa_qubo import (
    evaluate_qubo,
    fit_fixed_qubo,
)
from experiments.CIFAR10_experiments.cifar_dfa_ste_binary import (
    evaluate_ste_head,
)
from models.ls_heads import fit_ridge_classifier, predict_scores


class TestCIFAR10Heads(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        # Synthetic 128-dimensional representation
        self.H_train = torch.randn(50, 128)
        self.y_train = torch.randint(0, 10, (50,))
        self.H_val = torch.randn(15, 128)
        self.y_val = torch.randint(0, 10, (15,))
        self.H_test = torch.randn(20, 128)
        self.y_test = torch.randint(0, 10, (20,))

    def test_head1_ridge_least_squares(self):
        w, accs, fit_time = fit_and_evaluate_ridge(
            self.H_train, self.y_train, self.H_val, self.y_val, self.H_test, self.y_test
        )
        self.assertEqual(w.shape, (129, 10))
        scores = predict_scores(self.H_test, w)
        self.assertEqual(scores.shape, (20, 10))
        self.assertTrue(0.0 <= accs["test"] <= 100.0)

    def test_head2_binarized_ridge(self):
        ridge_w = fit_ridge_classifier(self.H_train, self.y_train, num_classes=10)
        (bin_w, scales, biases), accs, fit_time = evaluate_binarized_head(
            self.H_train, self.y_train, self.H_val, self.y_val, self.H_test, self.y_test, ridge_w
        )
        # Check weights are strictly bipolar {-1, +1}
        unique_w = torch.unique(bin_w)
        for u in unique_w:
            self.assertIn(u.item(), [-1.0, 1.0])
        self.assertEqual(bin_w.shape, (128, 10))
        self.assertEqual(scales.shape, (10,))
        self.assertEqual(biases.shape, (10,))

        scores = predict_binary_scores(self.H_test, bin_w, scales, biases)
        self.assertEqual(scores.shape, (20, 10))
        self.assertTrue(0.0 <= accs["test"] <= 100.0)

    def test_head3_direct_binary(self):
        (w_bin, bias), accs, fit_time = evaluate_direct_binary_head(
            self.H_train, self.y_train, self.H_val, self.y_val, self.H_test, self.y_test,
            steps=20, lr=0.05, seed=42
        )
        unique_w = torch.unique(w_bin)
        for u in unique_w:
            self.assertIn(u.item(), [-1.0, 1.0])
        self.assertEqual(w_bin.shape, (128, 10))
        self.assertEqual(bias.shape, (10,))

        scores = predict_direct_binary(self.H_test, w_bin, bias)
        self.assertEqual(scores.shape, (20, 10))
        self.assertTrue(0.0 <= accs["test"] <= 100.0)

    def test_head4_ste_binary(self):
        best, accs, fit_time = evaluate_ste_head(
            self.H_train, self.y_train, self.H_val, self.y_val, self.H_test, self.y_test,
            epochs=10, lr=0.01, seed=42
        )
        w_ste = best["weights"]
        bias_ste = best["bias"]
        unique_w = torch.unique(w_ste)
        for u in unique_w:
            self.assertIn(u.item(), [-1.0, 1.0])
        self.assertEqual(w_ste.shape, (128, 10))
        self.assertEqual(bias_ste.shape, (10,))
        self.assertTrue(0.0 <= accs["test"] <= 100.0)

    def test_head5_qubo_binary(self):
        w_bin, scales, biases, fit_time = fit_fixed_qubo(
            self.H_train, self.y_train, num_reads=5, num_sweeps=10, seed=42
        )
        unique_w = torch.unique(w_bin)
        for u in unique_w:
            self.assertIn(u.item(), [-1.0, 1.0])
        self.assertEqual(w_bin.shape, (128, 10))
        self.assertEqual(scales.shape, (10,))
        self.assertEqual(biases.shape, (10,))

        test_acc = evaluate_qubo(self.H_test, self.y_test, w_bin, scales, biases)
        self.assertTrue(0.0 <= test_acc <= 100.0)


if __name__ == "__main__":
    unittest.main()
