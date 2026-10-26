"""Synthetic tests for MNIST fixed-representation provenance and alignment."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch
from torch.utils.data import DataLoader

from experiments.MNIST_experiments.bnn_dfa_qubo import collect_hidden_features
from experiments.MNIST_experiments import mnist_representation as cache_manager
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
            self.assertIn("git_commit_sha", first["metadata"]["run_provenance"])
            self.assertIn("package_versions", first["metadata"]["run_provenance"])
            torch.testing.assert_close(first["H_train"], second["H_train"])
            incompatible = build_metadata(self.train, self.test, seed=5, epochs=2, validation_fraction=0.2)
            with self.assertRaisesRegex(ValueError, "metadata mismatch in:.*seed"):
                load_representation(path, incompatible, self.train, self.test)
            wrong_dataset = dict(metadata, dataset_fingerprint="0" * 64)
            with self.assertRaisesRegex(ValueError, "dataset_fingerprint"):
                load_representation(path, wrong_dataset, self.train, self.test)
            wrong_backbone = dict(metadata, backbone=dict(metadata["backbone"]))
            wrong_backbone["backbone"]["fc1"] = dict(metadata["backbone"]["fc1"], out_features=64)
            with self.assertRaisesRegex(ValueError, "backbone.fc1.out_features"):
                load_representation(path, wrong_backbone, self.train, self.test)
            wrong_training = dict(metadata, training=dict(metadata["training"], epochs=9))
            with self.assertRaisesRegex(ValueError, "training.epochs"):
                load_representation(path, wrong_training, self.train, self.test)
            with self.assertRaises(FileExistsError):
                save_representation(path, data, metadata)

    def _artifact_data_for(self, metadata):
        train_indices = metadata["split_indices"]["train"]
        val_indices = metadata["split_indices"]["validation"]
        test_indices = metadata["split_indices"]["test"]
        return {
            "H_train": torch.zeros(len(train_indices), 128),
            "y_train": self.train.targets[train_indices].clone(),
            "H_validation": torch.zeros(len(val_indices), 128),
            "y_validation": self.train.targets[val_indices].clone(),
            "H_test": torch.zeros(len(test_indices), 128),
            "y_test": self.test.targets[test_indices].clone(),
        }

    def _cache_path(self, output_dir, seed=4, epochs=2, fraction=0.2):
        return cache_manager.representation_path(output_dir, seed, epochs, fraction)

    def test_cache_manager_reuses_compatible_artifact_including_legacy_provenance_shape(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            output_dir = Path(temp)
            metadata = build_metadata(self.train, self.test, seed=4, epochs=2, validation_fraction=0.2)
            # This matches the prior artifact schema that triggered the onboarding failure:
            # provenance was absent, and the execution device is not a model compatibility key.
            metadata.pop("run_provenance")
            metadata["training"].pop("device")
            path = self._cache_path(output_dir)
            save_representation(path, self._artifact_data_for(metadata), metadata)
            with mock.patch.object(cache_manager, "train_dfa", side_effect=AssertionError("must reuse")):
                representation, loaded_path, source = cache_manager.load_or_train_mnist_representation(
                    self.train, self.test, seed=4, epochs=2, validation_fraction=0.2,
                    device="cpu", output_dir=output_dir,
                )
            self.assertEqual(loaded_path, path)
            self.assertEqual(source, "reused")
            self.assertEqual(representation["metadata"]["seed"], 4)

    def test_cache_manager_rejects_incompatible_artifact_with_field_and_recovery_command(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            output_dir = Path(temp)
            expected = build_metadata(self.train, self.test, seed=4, epochs=2, validation_fraction=0.2)
            incompatible = dict(expected)
            incompatible["training"] = dict(expected["training"], epochs=9)
            save_representation(self._cache_path(output_dir), self._artifact_data_for(expected), incompatible)
            with mock.patch.object(cache_manager, "train_dfa", side_effect=AssertionError("must stop")):
                with self.assertRaisesRegex(RuntimeError, "training.epochs.*representation-dir") as caught:
                    cache_manager.load_or_train_mnist_representation(
                        self.train, self.test, seed=4, epochs=2, validation_fraction=0.2,
                        device="cpu", output_dir=output_dir,
                    )
            self.assertIn("existing artifact was left unchanged", str(caught.exception))

    def test_cache_manager_trains_and_saves_when_artifact_is_missing(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            output_dir = Path(temp)

            def fake_collect(_model, loader, _device):
                subset = loader.dataset
                indices = torch.as_tensor(subset.indices, dtype=torch.long)
                labels = subset.dataset.targets[indices]
                return torch.zeros(len(indices), 128), labels

            with mock.patch.object(cache_manager, "BNN") as bnn, \
                 mock.patch.object(cache_manager, "DFAClassifier"), \
                 mock.patch.object(cache_manager.torch.optim, "Adam"), \
                 mock.patch.object(cache_manager, "train_dfa", return_value=(0.0, 0.0)) as trainer, \
                 mock.patch.object(cache_manager, "collect_hidden_features", side_effect=fake_collect):
                bnn.return_value.to.return_value = bnn.return_value
                representation, path, source = cache_manager.load_or_train_mnist_representation(
                    self.train, self.test, seed=4, epochs=1, validation_fraction=0.2,
                    device="cpu", output_dir=output_dir,
                )
            self.assertTrue(path.is_file())
            self.assertEqual(source, "trained")
            self.assertEqual(trainer.call_count, 1)
            self.assertEqual(len(representation["y_test"]), len(self.test))

    def test_corrupt_cache_stops_with_recovery_command_and_preserves_file(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            output_dir = Path(temp)
            path = self._cache_path(output_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            original = b"not a torch artifact"
            path.write_bytes(original)
            with mock.patch.object(cache_manager, "train_dfa", side_effect=AssertionError("must stop")):
                with self.assertRaisesRegex(RuntimeError, "could not be reused.*representation-dir"):
                    cache_manager.load_or_train_mnist_representation(
                        self.train, self.test, seed=4, epochs=2, validation_fraction=0.2,
                        device="cpu", output_dir=output_dir,
                    )
            self.assertEqual(path.read_bytes(), original)

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
