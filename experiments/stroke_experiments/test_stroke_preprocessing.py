"""Focused integrity tests: python -m unittest experiments.stroke_experiments.test_stroke_preprocessing"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.stroke_data import prepare_stroke_data, load_stroke_metadata
from utils.stroke_representation import save_representation, load_representation


class StrokePreprocessingTests(unittest.TestCase):
    @staticmethod
    def sample():
        # Enough positive cases for a stratified split; categorical levels also
        # exercise missing-value handling and the encoder's unknown-category path.
        n = 400
        return pd.DataFrame({
            "id": np.arange(n), "gender": ["Female", "Male"] * 200,
            "age": np.linspace(1, 82, n), "hypertension": [0, 1] * 200,
            "heart_disease": [0] * 390 + [1] * 10,
            "ever_married": ["No", "Yes"] * 200,
            "work_type": (["Private", "Govt_job", "children", "Self-employed"] * 100),
            "Residence_type": ["Urban", "Rural"] * 200,
            "avg_glucose_level": np.linspace(60, 250, n),
            "bmi": [np.nan if i % 19 == 0 else 20 + i % 20 for i in range(n)],
            "smoking_status": (["never smoked", "Unknown", "formerly smoked", "smokes"] * 100),
            "stroke": [1 if i % 10 == 0 else 0 for i in range(n)],
        })

    def test_split_schema_training_only_balance_and_saved_state(self):
        with tempfile.TemporaryDirectory(dir=Path("data")) as tmp:
            source = Path(tmp) / "stroke.csv"
            self.sample().to_csv(source, index=False)
            Xtr, Xv, Xt, ytr, yv, yt = prepare_stroke_data(csv_path=source, output_dir=tmp, seed=9)
            self.assertEqual(list(Xtr.columns), list(Xv.columns))
            self.assertEqual(list(Xtr.columns), list(Xt.columns))
            self.assertEqual(Xtr.shape[1], len(load_stroke_metadata(Path(tmp) / "metadata.json")["feature_names"]))
            self.assertTrue(np.isfinite(Xtr.to_numpy()).all())
            self.assertTrue(np.isfinite(Xv.to_numpy()).all())
            self.assertTrue(np.isfinite(Xt.to_numpy()).all())
            self.assertEqual(ytr.value_counts().nunique(), 1)
            self.assertEqual(int(yv.sum()), 6)
            self.assertEqual(int(yt.sum()), 6)
            meta = load_stroke_metadata(Path(tmp) / "metadata.json")
            self.assertEqual(meta["seed"], 9)
            self.assertEqual(len(meta["split_row_ids"]["train"]), 280)
            self.assertEqual(meta["balancing_method"], "random minority oversampling by exact row duplication")
            fitted = __import__("joblib").load(Path(tmp) / "preprocessor.joblib")
            unseen = self.sample().iloc[:2].copy()
            unseen["smoking_status"] = "future category"
            self.assertEqual(fitted.transform(unseen).shape[1], len(meta["feature_names"]))
            representation = {
                "seed": 4, "device": "cpu", "best_epoch": 1, "best_val_ba": .5,
                "feature_names": meta["feature_names"], "data_metadata": meta,
                "H_train": torch.zeros(len(ytr), 3), "H_val": torch.zeros(len(yv), 3), "H_test": torch.zeros(len(yt), 3),
                "y_train": ytr, "y_val": yv, "y_test": yt,
            }
            rep_path = Path(tmp) / "representation.pt"
            save_representation(representation, rep_path, meta["feature_names"])
            loaded = load_representation(rep_path)
            self.assertEqual(len(loaded["y_train"]), loaded["H_train"].shape[0])

    def test_unbalanced_mode_keeps_original_training_records(self):
        with tempfile.TemporaryDirectory(dir=Path("data")) as tmp:
            source = Path(tmp) / "stroke.csv"
            self.sample().to_csv(source, index=False)
            *_, ytr, _, _ = prepare_stroke_data(csv_path=source, output_dir=tmp, seed=9, balance=False)
            self.assertEqual(len(ytr), 280)
            self.assertGreater(ytr.value_counts().iloc[0], ytr.value_counts().iloc[1])


if __name__ == "__main__":
    unittest.main()
