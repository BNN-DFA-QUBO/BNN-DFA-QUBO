"""Comprehensive unit tests for the Bank dataset extension pipeline."""
import unittest
import torch
import numpy as np
from pathlib import Path

from models.bank_bnn import BankANN, BankBNN
from models.bank_dfa import BankDFA, BankDFAFunction
from utils.bank.data import NUM_FEATURES, NUM_CLASSES, BankDataset
from utils.bank.metrics import compute_classification_metrics, find_best_threshold
from utils.bank.representation import (
    build_metadata,
    dataset_fingerprint,
    save_representation,
    load_representation,
    validate_representation,
)
from models.ls_heads import fit_ridge_classifier, predict_scores
from experiments.bank_experiments.compare_ls_binary_qubo_shared import fit_binarized_ridge
from experiments.bank_experiments.tune_qubo_validation import (
    create_binary_targets,
    calculate_final_scale_and_bias,
    evaluate_binary_head,
)
from models.qubo_head import build_ising_model


class TestBankModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.hidden_size = 64
        self.dummy_x = torch.randn(self.batch_size, NUM_FEATURES)
        self.dummy_y = torch.randint(0, NUM_CLASSES, (self.batch_size,))

    def test_bank_ann_forward(self):
        model = BankANN(input_size=NUM_FEATURES, hidden_size=self.hidden_size, num_classes=NUM_CLASSES)
        out = model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, NUM_CLASSES))

        features = model.extract_features(self.dummy_x)
        self.assertEqual(features.shape, (self.batch_size, self.hidden_size))
        self.assertTrue((features >= 0).all())  # ReLU activation

    def test_bank_bnn_forward(self):
        model = BankBNN(input_size=NUM_FEATURES, hidden_size=self.hidden_size, num_classes=NUM_CLASSES)
        out = model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, NUM_CLASSES))

        features = model.extract_features(self.dummy_x)
        self.assertEqual(features.shape, (self.batch_size, self.hidden_size))
        self.assertTrue((features >= 0).all())

    def test_bank_dfa_gradients(self):
        model = BankBNN(input_size=NUM_FEATURES, hidden_size=self.hidden_size, num_classes=NUM_CLASSES)
        dfa = BankDFA(hidden_size=self.hidden_size, output_size=NUM_CLASSES, device="cpu")

        self.assertEqual(dfa.feedback.shape, (NUM_CLASSES, self.hidden_size))

        hidden = model.fc1(self.dummy_x)
        hidden_relu = torch.relu(hidden)
        outputs = model.fc2(hidden_relu)

        output_error = BankDFAFunction.output_error(outputs, self.dummy_y)
        self.assertEqual(output_error.shape, (self.batch_size, NUM_CLASSES))

        hidden_error = dfa.hidden_gradient(output_error)
        self.assertEqual(hidden_error.shape, (self.batch_size, self.hidden_size))


class TestBankMetrics(unittest.TestCase):
    def test_metrics_calculation(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.3, 0.7, 0.6])

        metrics = compute_classification_metrics(y_true, y_pred, y_scores)
        self.assertIn("accuracy", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertGreater(metrics["accuracy"], 50.0)

    def test_threshold_tuning(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        thresh = find_best_threshold(y_true, scores)
        self.assertGreater(thresh, 0.2)
        self.assertLess(thresh, 0.8)


class TestBankRepresentation(unittest.TestCase):
    def test_dataset_fingerprint_and_metadata(self):
        ds_tr = BankDataset(np.random.randn(20, NUM_FEATURES).astype(np.float32), np.random.randint(0, 2, 20))
        ds_val = BankDataset(np.random.randn(10, NUM_FEATURES).astype(np.float32), np.random.randint(0, 2, 10))
        ds_te = BankDataset(np.random.randn(10, NUM_FEATURES).astype(np.float32), np.random.randint(0, 2, 10))

        fp1 = dataset_fingerprint(ds_tr, ds_val, ds_te)
        fp2 = dataset_fingerprint(ds_tr, ds_val, ds_te)
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)

        meta = build_metadata(ds_tr, ds_val, ds_te, seed=42, epochs=5, hidden_size=64)
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["epochs"], 5)
        self.assertEqual(meta["hidden_size"], 64)

    def test_save_load_validation(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir) / "test_repr.pt"
            meta = {"version": "bank-bnn-dfa-representation-v1", "seed": 42, "epochs": 5, "hidden_size": 32}
            payload = {
                "H_train": torch.randn(20, 32),
                "y_train": torch.randint(0, 2, (20,)),
                "H_val": torch.randn(10, 32),
                "y_val": torch.randint(0, 2, (10,)),
                "H_test": torch.randn(10, 32),
                "y_test": torch.randint(0, 2, (10,)),
            }
            save_representation(tmp_p, payload, meta)
            loaded = load_representation(tmp_p, meta)
            self.assertEqual(loaded["H_train"].shape, (20, 32))
            validate_representation(loaded, meta)


class TestBankHeadsAndQUBO(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.n_samples = 30
        self.hidden_dim = 16
        self.H = torch.randn(self.n_samples, self.hidden_dim)
        self.y = torch.randint(0, NUM_CLASSES, (self.n_samples,))

    def test_real_ridge_and_binarized_ridge(self):
        ridge_w = fit_ridge_classifier(self.H, self.y, num_classes=NUM_CLASSES)
        self.assertEqual(ridge_w.shape, (self.hidden_dim + 1, NUM_CLASSES))

        scores = predict_scores(self.H, ridge_w)
        self.assertEqual(scores.shape, (self.n_samples, NUM_CLASSES))

        w_bin, sc, b = fit_binarized_ridge(self.H, self.y, ridge_w, num_classes=NUM_CLASSES)
        self.assertEqual(w_bin.shape, (self.hidden_dim, NUM_CLASSES))
        self.assertTrue(torch.all(torch.abs(w_bin) == 1.0))
        self.assertEqual(sc.shape, (NUM_CLASSES,))
        self.assertEqual(b.shape, (NUM_CLASSES,))

        bin_scores = evaluate_binary_head(self.H, w_bin, sc, b)
        self.assertEqual(bin_scores.shape, (self.n_samples, NUM_CLASSES))

    def test_ising_formulation(self):
        target = torch.where(self.y == 1, 1.0, -1.0)
        h, J = build_ising_model(self.H, target, alpha=0.5)
        self.assertEqual(len(h), self.hidden_dim)
        # J should contain pairs
        self.assertGreater(len(J), 0)


if __name__ == "__main__":
    unittest.main()

