"""Synthetic tests for MNIST fixed-representation provenance and alignment."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch.utils.data import DataLoader

from experiments.MNIST_experiments.bnn_dfa_qubo import collect_hidden_features
from utils.MNIST.representation import (
    build_metadata,
    load_representation,
    save_representation,
)


class TinyDataset:
    def __init__(self, labels):
        self.targets = torch.tensor(labels, dtype=torch.long)
        self.data = torch.arange(len(labels) * 4, dtype=torch.uint8).reshape(len(labels), 2, 2)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index].float().reshape(1, 2, 2), self.targets[index]


class TinyFeatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 128)


class MnistRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.train = TinyDataset([0, 1] * 20)
        self.test = TinyDataset([0, 1] * 5)

    def test_repeated_feature_extraction_is_deterministic_and_label_aligned(self):
        torch.manual_seed(17)
        model = TinyFeatureModel().eval()
        loader = DataLoader(self.train, batch_size=7, shuffle=False)
        first_h, first_y = collect_hidden_features(model, loader, "cpu")
        second_h, second_y = collect_hidden_features(model, loader, "cpu")
        torch.testing.assert_close(first_h, second_h, rtol=0, atol=0)
        torch.testing.assert_close(first_y, second_y, rtol=0, atol=0)
        torch.testing.assert_close(first_y, self.train.targets)

    def test_cache_loads_identical_artifact_and_rejects_incompatible_seed(self):
        metadata = build_metadata(self.train, self.test, seed=4, epochs=2, validation_fraction=0.2)
        n_train = metadata["sample_counts"]["train"]
        n_val = metadata["sample_counts"]["validation"]
        train_labels = self.train.targets[metadata["split_indices"]["train"]]
        val_labels = self.train.targets[metadata["split_indices"]["validation"]]
        test_labels = self.test.targets
        data = {
            "H_train": torch.zeros(n_train, 128), "y_train": train_labels,
            "H_validation": torch.zeros(n_val, 128), "y_validation": val_labels,
            "H_test": torch.zeros(len(self.test), 128), "y_test": test_labels,
        }
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "representation.pt"
            saved = save_representation(path, data, metadata)
            first = load_representation(path, metadata, self.train, self.test)
            second = load_representation(path, metadata, self.train, self.test)
            self.assertEqual(first["artifact_id"], second["artifact_id"])
            self.assertEqual(first["artifact_id"], saved["artifact_id"])
            torch.testing.assert_close(first["H_train"], second["H_train"])
            incompatible = build_metadata(self.train, self.test, seed=5, epochs=2, validation_fraction=0.2)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_representation(path, incompatible, self.train, self.test)
            with self.assertRaises(FileExistsError):
                save_representation(path, data, metadata)

    def test_saving_misaligned_labels_is_rejected_on_load(self):
        metadata = build_metadata(self.train, self.test, seed=8, epochs=1, validation_fraction=0.2)
        train_ids = metadata["split_indices"]["train"]
        val_ids = metadata["split_indices"]["validation"]
        data = {
            "H_train": torch.zeros(len(train_ids), 128),
            "y_train": self.train.targets[train_ids].flip(0),
            "H_validation": torch.zeros(len(val_ids), 128),
            "y_validation": self.train.targets[val_ids],
            "H_test": torch.zeros(len(self.test), 128), "y_test": self.test.targets,
        }
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "misaligned.pt"
            save_representation(path, data, metadata)
            with self.assertRaisesRegex(ValueError, "misaligned"):
                load_representation(path, metadata, self.train, self.test)


if __name__ == "__main__":
    unittest.main()
