"""Configuration compatibility tests for saved Stroke representations."""
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

from utils.stroke_config import PREPROCESSING_VERSION
from utils.stroke_representation import (
    load_representation,
    load_stroke_head_representation,
    save_representation,
)


def make_artifact(root, include_training_config=True):
    metadata = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset_sha256": "d" * 64,
        "split_seed": 17,
        "preprocessing_config_hash": "c" * 64,
        "feature_names": ["age"],
        "input_size": 1,
        "balancing_method": "none",
        "original_train_class_counts": {"0": 2, "1": 2},
        "resampled_train_class_counts": {"0": 2, "1": 2},
        "validation_class_counts": {"0": 1, "1": 1},
        "test_class_counts": {"0": 1, "1": 1},
        "resampled_train_row_ids": [0, 1, 2, 3],
        "split_row_ids": {"train": [0, 1, 2, 3], "validation": [4, 5], "test": [6, 7]},
        "split_row_labels": {"train": [0, 1, 0, 1], "validation": [0, 1], "test": [0, 1]},
    }
    result = {
        "seed": 42,
        "training_seed": 42,
        "device": "cpu",
        "best_epoch": 5,
        "best_val_ba": 0.75,
        "feature_names": ["age"],
        "data_metadata": metadata,
        "H_train": torch.arange(4 * 64, dtype=torch.float32).reshape(4, 64),
        "H_val": torch.zeros(2, 64),
        "H_test": torch.ones(2, 64),
        "y_train": pd.Series([0, 1, 0, 1]),
        "y_val": pd.Series([0, 1]),
        "y_test": pd.Series([0, 1]),
    }
    if include_training_config:
        result["training_config"] = {
            "hidden_size": 64, "batch_size": 64, "epochs": 50,
            "learning_rate": 0.001,
            "loss": "BCEWithLogitsLoss(reduction='mean', pos_weight=None)",
        }
    path = root / "stroke_representation.pt"
    saved = save_representation(result, path, ["age"])
    return path, metadata, saved


class StrokeRepresentationCompatibilityTests(unittest.TestCase):
    def test_matching_seed_dimensions_preprocessing_and_training_config_load(self):
        expected_config = {
            "hidden_size": 64, "batch_size": 64, "epochs": 50,
            "learning_rate": 0.001,
            "loss": "BCEWithLogitsLoss(reduction='mean', pos_weight=None)",
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / "data") as temp:
            path, metadata, saved = make_artifact(Path(temp))
            loaded = load_representation(
                path, expected_metadata=metadata, expected_seed=42,
                expected_hidden_size=64, expected_training_config=expected_config,
            )
            self.assertEqual(loaded["representation_sha256"], saved["representation_sha256"])
            self.assertEqual(loaded["integrity_status"], "content_and_row_id_verified")

    def test_incompatible_seed_hidden_size_training_or_split_are_rejected(self):
        expected_config = {"epochs": 50}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / "data") as temp:
            path, metadata, _ = make_artifact(Path(temp))
            cases = (
                ({"expected_seed": 99}, "training seed"),
                ({"expected_hidden_size": 32}, "hidden dimension"),
                ({"expected_training_config": {"epochs": 20}}, "training configuration field"),
            )
            for options, message in cases:
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                    load_representation(path, expected_metadata=metadata, **options)
            wrong_split = dict(metadata, split_seed=18)
            with self.assertRaisesRegex(ValueError, "split_seed"):
                load_representation(path, expected_metadata=wrong_split,
                                    expected_training_config=expected_config)

    def test_missing_optional_config_warns_but_legacy_remains_explicitly_unverified(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / "data") as temp:
            path, metadata, _ = make_artifact(Path(temp), include_training_config=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = load_representation(
                    path, expected_metadata=metadata, expected_seed=42,
                    expected_hidden_size=64, expected_training_config={"epochs": 50},
                )
            self.assertEqual(loaded["integrity_status"], "content_and_row_id_verified")
            self.assertTrue(any("configuration compatibility could not be established" in str(w.message)
                                for w in caught))

            legacy = torch.load(path, weights_only=False)
            legacy["representation_version"] = "stroke-representation-v2"
            legacy.pop("representation_sha256")
            legacy.pop("integrity_status")
            torch.save(legacy, path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = load_representation(path, expected_metadata=metadata,
                                             expected_hidden_size=64,
                                             expected_training_config={"epochs": 50})
            self.assertEqual(loaded["integrity_status"], "legacy_unverified")
            self.assertTrue(any("legacy Stroke representation" in str(w.message) for w in caught))

    def test_standalone_head_loader_enforces_known_hidden_size_without_using_head_seed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / "data") as temp:
            path, _, _ = make_artifact(Path(temp))
            with mock.patch("utils.stroke_representation.METADATA_PATH", Path(temp) / "missing-metadata.json"), \
                 mock.patch("utils.stroke_representation.RAW_PATH", Path(temp) / "missing-source.csv"), \
                 warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = load_stroke_head_representation(path)
            self.assertEqual(loaded["hidden_size"], 64)
            self.assertEqual(loaded["training_seed"], 42)
            self.assertTrue(any("No current Stroke preprocessing metadata" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
