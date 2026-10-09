"""Scientific invariant tests: python -m unittest experiments.stroke_experiments.test_stroke_preprocessing"""
import tempfile
import unittest
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from utils.stroke_config import PREPROCESSING_VERSION
from utils.stroke_data import StrokePreprocessor, _split, _validate, prepare_stroke_data, load_stroke_metadata
from utils.stroke_representation import save_representation, load_representation
from models.stroke_dfa import StrokeDFAFunction
from experiments.stroke_experiments.stroke_dfa_least_squares import fit_signed_least_squares


class StrokePreprocessingTests(unittest.TestCase):
    @staticmethod
    def sample():
        n = 400
        df = pd.DataFrame({
            "id": np.arange(n), "gender": ["Female", "Male"] * 200,
            "age": np.linspace(1, 82, n), "hypertension": [0, 1] * 200,
            "heart_disease": [0] * 390 + [1] * 10,
            "ever_married": ["No", "Yes"] * 200,
            "work_type": ["Private", "Govt_job", "children", "Self-employed"] * 100,
            "Residence_type": ["Urban", "Rural"] * 200,
            "avg_glucose_level": np.linspace(60, 250, n),
            "bmi": [np.nan if i % 19 == 0 else 20 + i % 20 for i in range(n)],
            "smoking_status": ["never smoked", "Unknown", "formerly smoked", "smokes"] * 100,
            "stroke": [1 if i % 10 == 0 else 0 for i in range(n)],
        })
        other = df.iloc[[0]].copy()
        other["id"] = 400
        other["gender"] = "Other"
        other["stroke"] = 0
        return pd.concat([df, other], ignore_index=True)

    def test_required_columns_and_binary_target_validation(self):
        df = self.sample()
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            _validate(df.drop(columns="bmi"))
        bad = df.copy()
        bad.loc[0, "stroke"] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            _validate(bad)

    def test_deterministic_stratified_split_and_excluded_gender_provenance(self):
        df = self.sample().iloc[:-1].copy()
        a = _split(df, 19)
        b = _split(df, 19)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        train, val, test = a
        self.assertEqual((len(train), len(val), len(test)), (280, 60, 60))
        self.assertAlmostEqual(df.stroke.iloc[train].mean(), df.stroke.mean(), delta=.02)
        self.assertFalse(set(train) & set(val) or set(train) & set(test) or set(val) & set(test))

    def test_engineering_matches_reference_features_and_categorical_grouping(self):
        df = self.sample().iloc[:320].copy()
        pre = StrokePreprocessor().fit(df)
        transformed = pre.transform(df.iloc[[1, 2, 3]])
        self.assertEqual(pre.feature_names[:12], ["gender", "age", "hypertension", "heart_disease", "ever_married", "Residence_type",
                                                   "avg_glucose_level", "bmi", "cardio_comorbidity_index", "age_glucose_interaction",
                                                   "glucose_risk_tier", "metabolic_syndrome_flag"])
        self.assertEqual(len(pre.feature_names), 23)
        self.assertIn("work_type_Govt_job", transformed.columns)
        self.assertIn("smoking_never smoked", transformed.columns)
        row = df.iloc[1]
        self.assertAlmostEqual(transformed.iloc[0]["age_glucose_interaction"], row.age * row.avg_glucose_level, delta=1e-5)
        self.assertEqual(transformed.iloc[0]["glucose_risk_tier"], 0)
        child = df.iloc[[2]].copy()
        never = df.iloc[[0]].copy()
        child["work_type"] = "children"
        never["work_type"] = "Never_worked"
        child_features, never_features = pre.transform(child), pre.transform(never)
        work_columns = [name for name in pre.feature_names if name.startswith("work_type_")]
        np.testing.assert_array_equal(child_features[work_columns].to_numpy(), never_features[work_columns].to_numpy())
        self.assertEqual(int(transformed.iloc[1]["cardio_comorbidity_index"]),
                         int(transformed.iloc[1]["hypertension"] + transformed.iloc[1]["heart_disease"]))
        clinical = df.iloc[[10]].copy()
        clinical["hypertension"] = 1
        clinical["heart_disease"] = 1
        clinical["avg_glucose_level"] = 160
        clinical["bmi"] = 35
        engineered = pre.transform(clinical).iloc[0]
        self.assertEqual(engineered["cardio_comorbidity_index"], 2)
        self.assertEqual(engineered["glucose_risk_tier"], 2)
        self.assertEqual(engineered["metabolic_syndrome_flag"], 1)

    def test_fit_is_train_only_unknown_categories_and_power_transform(self):
        df = self.sample().iloc[:320].copy()
        train, heldout = df.iloc[:260].copy(), df.iloc[260:].copy()
        pre = StrokePreprocessor().fit(train)
        lambdas = pre.power.lambdas_.copy()
        heldout.loc[:, "avg_glucose_level"] = 10000
        heldout.loc[:, "bmi"] = 500
        heldout.loc[:, "work_type"] = "future work category"
        heldout.loc[:, "smoking_status"] = "future smoking category"
        output = pre.transform(heldout)
        self.assertTrue(np.isfinite(output.to_numpy()).all())
        np.testing.assert_array_equal(lambdas, pre.power.lambdas_)
        self.assertNotIn("future work category", pre.encoder.categories_[0])
        self.assertNotIn("future smoking category", pre.encoder.categories_[1])

    def test_balance_only_train_counts_provenance_and_reproducibility(self):
        with tempfile.TemporaryDirectory(dir=Path("data")) as tmp:
            root = Path(tmp)
            source = root / "stroke.csv"
            self.sample().to_csv(source, index=False)
            balanced = prepare_stroke_data(csv_path=source, output_dir=root / "balanced", split_seed=9)
            metadata = load_stroke_metadata(root / "balanced" / "metadata.json")
            self.assertEqual(metadata["split_policy"], "stratified_train_test_split_70_15_15_v1")
            self.assertEqual(metadata["preprocessing_version"], PREPROCESSING_VERSION)
            self.assertEqual(metadata["split_sizes"], {"train": 280, "validation": 60, "test": 60})
            self.assertEqual(metadata["balancing_method"], "random_oversample")
            self.assertEqual(metadata["original_train_class_counts"], {"0": 252, "1": 28})
            self.assertEqual(metadata["resampled_train_class_counts"], {"0": 252, "1": 252})
            self.assertEqual(metadata["validation_class_counts"], {"0": 54, "1": 6})
            self.assertEqual(metadata["test_class_counts"], {"0": 54, "1": 6})
            ids = metadata["split_row_ids"]
            self.assertEqual(set(ids["train"]) | set(ids["validation"]) | set(ids["test"]), set(range(400)))
            self.assertTrue(all(not set(ids[a]) & set(ids[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))))
            self.assertEqual(len(balanced[3]), len(metadata["resampled_train_row_ids"]))
            unbalanced = prepare_stroke_data(csv_path=source, output_dir=root / "unbalanced", split_seed=9, balance_method="none")
            np.testing.assert_array_equal(balanced[1], unbalanced[1])
            np.testing.assert_array_equal(balanced[2], unbalanced[2])
            np.testing.assert_array_equal(balanced[4], unbalanced[4])
            np.testing.assert_array_equal(balanced[5], unbalanced[5])
            repeat = prepare_stroke_data(csv_path=source, output_dir=root / "balanced", split_seed=9)
            np.testing.assert_array_equal(balanced[0].to_numpy(), repeat[0].to_numpy())
            np.testing.assert_array_equal(balanced[3].to_numpy(), repeat[3].to_numpy())
            self.assertEqual(len(ids["train"]), 280)  # excluded Other row 400 never enters a split

    def test_stale_preprocessing_artifacts_require_explicit_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path("data")) as tmp:
            root = Path(tmp)
            source = root / "stroke.csv"
            self.sample().iloc[:-1].to_csv(source, index=False)
            prepare_stroke_data(csv_path=source, output_dir=root / "artifacts", split_seed=7)
            modified = self.sample().iloc[:-2].copy()
            modified.to_csv(source, index=False)
            with self.assertRaisesRegex(FileExistsError, "stale or incompatible"):
                prepare_stroke_data(csv_path=source, output_dir=root / "artifacts", split_seed=7)
            regenerated = prepare_stroke_data(csv_path=source, output_dir=root / "artifacts", split_seed=7, overwrite=True)
            new_metadata = load_stroke_metadata(root / "artifacts" / "metadata.json")
            self.assertEqual(sum(new_metadata["split_sizes"].values()), 399)

    def test_matching_artifacts_reject_tampered_split_provenance(self):
        with tempfile.TemporaryDirectory(dir=Path("data")) as tmp:
            root = Path(tmp)
            source = root / "stroke.csv"
            artifacts = root / "artifacts"
            self.sample().to_csv(source, index=False)
            prepare_stroke_data(csv_path=source, output_dir=artifacts, split_seed=17)
            metadata_path = artifacts / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["split_policy"] = "tampered_policy"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "split_policy"):
                prepare_stroke_data(csv_path=source, output_dir=artifacts, split_seed=17)
            # Refusal preserves both artifacts until an explicit overwrite is requested.
            self.assertEqual(json.loads(metadata_path.read_text())["split_policy"], "tampered_policy")

    def test_process_stroke_reports_selected_artifact_directory(self):
        from experiments.stroke_experiments import process_stroke
        with tempfile.TemporaryDirectory(dir=Path("data")) as tmp:
            output_dir = Path(tmp) / "custom-artifacts"
            X = pd.DataFrame([[0.0]], columns=["feature"])
            y = pd.Series([0])
            metadata = {"feature_names": ["feature"], "resampled_train_class_counts": {"0": 1},
                        "validation_class_counts": {"0": 1}, "test_class_counts": {"0": 1},
                        "balancing_method": "none"}
            with patch("sys.argv", ["process_stroke", "--output-dir", str(output_dir)]), \
                 patch.object(process_stroke, "prepare_stroke_data", return_value=(X, X, X, y, y, y)) as prepare, \
                 patch.object(process_stroke, "load_stroke_metadata", return_value=metadata), \
                 redirect_stdout(io.StringIO()) as output:
                process_stroke.main()
            self.assertEqual(prepare.call_args.kwargs["output_dir"], output_dir)
            self.assertIn(str(output_dir), output.getvalue())

    def test_saved_representation_rejects_old_schema_and_misaligned_labels(self):
        metadata = {"preprocessing_version": PREPROCESSING_VERSION, "dataset_sha256": "a" * 64, "split_seed": 7,
                    "preprocessing_config_hash": "b" * 64, "feature_names": ["age"], "input_size": 1,
                    "resampled_train_class_counts": {"0": 2, "1": 2},
                    "validation_class_counts": {"0": 1, "1": 1}, "test_class_counts": {"0": 1, "1": 1},
                    "resampled_train_row_ids": list(range(4)),
                    "split_row_ids": {"train": [0, 1, 2, 3], "validation": [4, 5], "test": [6, 7]}}
        with tempfile.TemporaryDirectory(dir=Path("data")) as tmp:
            result = {"seed": 4, "training_seed": 4, "device": "cpu", "best_epoch": 1, "best_val_ba": .5,
                      "feature_names": ["age"], "data_metadata": metadata,
                      "H_train": torch.zeros(4, 3), "H_val": torch.zeros(2, 3), "H_test": torch.zeros(2, 3),
                      "y_train": pd.Series([0, 1, 0, 1]), "y_val": pd.Series([0, 1]), "y_test": pd.Series([0, 1])}
            path = Path(tmp) / "representation.pt"
            save_representation(result, path, ["age"])
            loaded = load_representation(path, expected_metadata=metadata)
            self.assertEqual(len(loaded["y_train"]), loaded["H_train"].shape[0])
            raw = torch.load(path, weights_only=False)
            raw["representation_version"] = "old"
            torch.save(raw, path)
            with self.assertRaisesRegex(ValueError, "stale"):
                load_representation(path, expected_metadata=metadata)
            raw["representation_version"] = "stroke-representation-v2"
            raw["y_val"] = torch.tensor([0.0])
            torch.save(raw, path)
            with self.assertRaisesRegex(ValueError, "do not align"):
                load_representation(path, expected_metadata=metadata)

    def test_dfa_output_gradient_matches_mean_unweighted_bce(self):
        logits = torch.tensor([-1.2, .1, 2.3], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 1.0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        expected, = torch.autograd.grad(loss, logits)
        actual = StrokeDFAFunction.output_error(logits, labels) / len(labels)
        torch.testing.assert_close(actual, expected)

    def test_rank_deficient_signed_least_squares_is_finite(self):
        base = torch.arange(12, dtype=torch.float32).unsqueeze(1)
        hidden = torch.cat((base, base, torch.ones_like(base)), dim=1)
        labels = torch.tensor([0, 1] * 6, dtype=torch.float32)
        weights = fit_signed_least_squares(hidden, labels)
        self.assertTrue(torch.isfinite(weights).all())

    def test_all_five_classifier_heads_run_from_one_valid_representation(self):
        from experiments.stroke_experiments.stroke_real_ls import train_real_ls
        from experiments.stroke_experiments.stroke_dfa_binarized_ls import train_binarized_ls
        from experiments.stroke_experiments.stroke_dfa_direct_binary import train_direct_binary
        from experiments.stroke_experiments.stroke_dfa_ste_binary import train_ste_binary
        from experiments.stroke_experiments.stroke_dfa_qubo import train_qubo
        from experiments.stroke_experiments.stroke_dfa_least_squares import train_signed_least_squares

        rng = np.random.default_rng(5)
        y_train = torch.tensor([0, 1] * 24, dtype=torch.float32)
        y_val = torch.tensor([0, 1] * 10, dtype=torch.float32)
        y_test = torch.tensor([0, 1] * 10, dtype=torch.float32)
        def hidden(labels):
            signal = labels.numpy()[:, None] * 0.7
            return torch.tensor(rng.normal(size=(len(labels), 6)) + signal, dtype=torch.float32)
        metadata = {
            "preprocessing_version": PREPROCESSING_VERSION, "dataset_sha256": "c" * 64, "split_seed": 11,
            "preprocessing_config_hash": "d" * 64, "input_size": 23, "feature_names": [f"x{i}" for i in range(23)],
            "balancing_method": "random_oversample", "original_train_class_counts": {"0": 24, "1": 24},
            "resampled_train_class_counts": {"0": 24, "1": 24}, "validation_class_counts": {"0": 10, "1": 10},
            "test_class_counts": {"0": 10, "1": 10}, "resampled_train_row_ids": list(range(48)),
            "split_row_ids": {"train": list(range(48)), "validation": list(range(48, 68)), "test": list(range(68, 88))},
        }
        with tempfile.TemporaryDirectory(dir=Path("data")) as tmp:
            path = Path(tmp) / "heads.pt"
            result = {"seed": 3, "training_seed": 3, "device": "cpu", "best_epoch": 1, "best_val_ba": .5,
                      "feature_names": metadata["feature_names"], "data_metadata": metadata,
                      "H_train": hidden(y_train), "H_val": hidden(y_val), "H_test": hidden(y_test),
                      "y_train": pd.Series(y_train.numpy()), "y_val": pd.Series(y_val.numpy()), "y_test": pd.Series(y_test.numpy())}
            save_representation(result, path, metadata["feature_names"])
            with patch("utils.stroke_representation.METADATA_PATH", Path(tmp) / "no_current_metadata.json"), \
                 patch("utils.stroke_representation.RAW_PATH", Path(tmp) / "no_raw.csv"), \
                 patch("experiments.stroke_experiments.stroke_dfa_ste_binary.STE_EPOCHS", 3), \
                 patch("experiments.stroke_experiments.stroke_dfa_qubo.QUBO_NUM_READS", 2):
                results = [train_real_ls(path), train_binarized_ls(path), train_direct_binary(path, 17),
                           train_ste_binary(path, 17), train_qubo(path, 17), train_signed_least_squares(path)]
            for head_result in results:
                self.assertTrue(np.isfinite(head_result["balanced_accuracy"]))
                self.assertEqual(head_result["tn"] + head_result["fp"] + head_result["fn"] + head_result["tp"], 20)

    def test_bnn_backprop_and_dfa_complete_one_epoch_with_provenance(self):
        from experiments.stroke_experiments import stroke_bnn_bp, stroke_bnn_dfa

        rng = np.random.default_rng(21)
        def split(n):
            labels = pd.Series([0, 1] * (n // 2), dtype=int)
            values = rng.normal(size=(n, 23)).astype(np.float32)
            values[:, 0] += labels.to_numpy() * .25
            return pd.DataFrame(values), labels
        X_train, y_train = split(80)
        X_val, y_val = split(40)
        X_test, y_test = split(40)
        prepared = (X_train, X_val, X_test, y_train, y_val, y_test)
        metadata = {
            "split_seed": 13, "preprocessing_version": PREPROCESSING_VERSION,
            "preprocessing_config_hash": "unit-test-config", "dataset_sha256": "unit-test-hash",
            "balancing_method": "random_oversample", "feature_names": list(X_train.columns),
            "original_train_class_counts": {"0": 40, "1": 40}, "resampled_train_class_counts": {"0": 40, "1": 40},
            "validation_class_counts": {"0": 20, "1": 20}, "test_class_counts": {"0": 20, "1": 20},
        }
        with tempfile.TemporaryDirectory(dir=Path("data")) as tmp:
            with patch.object(stroke_bnn_bp, "prepare_stroke_data", return_value=prepared), \
                 patch.object(stroke_bnn_bp, "load_stroke_metadata", return_value=metadata), \
                 patch.object(stroke_bnn_bp, "get_device", return_value=torch.device("cpu")), \
                 patch.object(stroke_bnn_bp, "TRAIN_EPOCHS", 1), patch.object(stroke_bnn_bp, "BATCH_SIZE", 16):
                bp = stroke_bnn_bp.train_bnn_bp(31, split_seed=13, output_dir=tmp)
            with patch.object(stroke_bnn_dfa, "prepare_stroke_data", return_value=prepared), \
                 patch.object(stroke_bnn_dfa, "load_stroke_metadata", return_value=metadata), \
                 patch.object(stroke_bnn_dfa, "get_device", return_value=torch.device("cpu")), \
                 patch.object(stroke_bnn_dfa, "TRAIN_EPOCHS", 1), patch.object(stroke_bnn_dfa, "BATCH_SIZE", 16):
                dfa = stroke_bnn_dfa.train_dfa(37, split_seed=13, output_dir=tmp)
            self.assertEqual(bp["H_train"].shape[0], len(bp["y_train"]))
            self.assertEqual(dfa["H_train"].shape[0], len(dfa["y_train"]))
            for path, training_seed in ((bp["metrics_path"], 31), (Path(tmp) / "stroke_bnn_dfa_seed_37.json", 37)):
                result_metadata = json.loads(Path(path).read_text())
                self.assertEqual(result_metadata["training_seed"], training_seed)
                self.assertEqual(result_metadata["split_seed"], 13)
                self.assertEqual(result_metadata["balancing_method"], "random_oversample")
                self.assertIn("pos_weight=None", result_metadata["loss"])

    def test_controlled_result_rows_record_all_seed_and_preprocessing_fields(self):
        from experiments.stroke_experiments import run_stroke_controlled
        with tempfile.TemporaryDirectory(dir=Path("data")) as tmp:
            result_path = Path(tmp) / "results.csv"
            row = {"method": "Real LS", "seed": 42, "model_seed": 42, "split_seed": 13,
                   "representation_seed": 42, "head_seed": "", "preprocessing_version": PREPROCESSING_VERSION,
                   "experiment_type": "controlled_end_to_end_head_comparison",
                   "method_variant": "continuous_least_squares", "objective": "squared residual error",
                   "target_encoding": "stroke labels {0,1}", "bias_treatment": "fitted intercept",
                   "optimization_procedure": "numpy.linalg.lstsq",
                   "preprocessing_config_hash": "config", "dataset_sha256": "hash",
                   "balancing_method": "random_oversample", "loss_configuration": "least squares",
                   "split_class_counts": '{"validation":{"0": 2, "1": 1}}', "balanced_accuracy": .5}
            with patch.object(run_stroke_controlled, "RESULTS_PATH", result_path):
                run_stroke_controlled.save_results([row])
            saved = pd.read_csv(result_path).iloc[0]
            self.assertEqual(saved["model_seed"], 42)
            self.assertEqual(saved["split_seed"], 13)
            self.assertEqual(saved["representation_seed"], 42)
            self.assertEqual(saved["preprocessing_version"], PREPROCESSING_VERSION)
            self.assertEqual(saved["experiment_type"], "controlled_end_to_end_head_comparison")
            self.assertEqual(saved["target_encoding"], "stroke labels {0,1}")

    def test_controlled_and_multiseed_method_variants_describe_distinct_objectives(self):
        from experiments.stroke_experiments.run_stroke_controlled import CONTROLLED_METHOD_METADATA
        from experiments.stroke_experiments.stroke_binary_head_multiseed import MULTISEED_METHOD_METADATA
        self.assertNotEqual(CONTROLLED_METHOD_METADATA["Direct Binary"]["method_variant"],
                            MULTISEED_METHOD_METADATA["Direct Binary"]["method_variant"])
        self.assertEqual(CONTROLLED_METHOD_METADATA["Direct Binary"]["bias_treatment"], "fixed at zero")
        self.assertEqual(MULTISEED_METHOD_METADATA["Direct Binary"]["objective"], "binary_cross_entropy_with_logits")
        self.assertNotEqual(CONTROLLED_METHOD_METADATA["STE Binary"]["objective"],
                            MULTISEED_METHOD_METADATA["STE Binary"]["objective"])
        self.assertEqual(CONTROLLED_METHOD_METADATA["STE Binary"]["target_encoding"],
                         "stroke 0 -> -1; stroke 1 -> +1")
        self.assertEqual(MULTISEED_METHOD_METADATA["STE Binary"]["target_encoding"], "stroke labels {0,1}")

    def test_end_to_end_synthetic_csv_through_models_and_all_heads(self):
        from experiments.stroke_experiments import stroke_bnn_bp, stroke_bnn_dfa
        from experiments.stroke_experiments.stroke_real_ls import train_real_ls
        from experiments.stroke_experiments.stroke_dfa_binarized_ls import train_binarized_ls
        from experiments.stroke_experiments.stroke_dfa_direct_binary import train_direct_binary
        from experiments.stroke_experiments.stroke_dfa_ste_binary import train_ste_binary
        from experiments.stroke_experiments.stroke_dfa_qubo import train_qubo
        from experiments.stroke_experiments.stroke_dfa_least_squares import train_signed_least_squares

        with tempfile.TemporaryDirectory(dir=Path("data")) as tmp:
            root = Path(tmp)
            source = root / "synthetic-stroke.csv"
            artifacts = root / "preprocessing"
            output = root / "runs"
            self.sample().to_csv(source, index=False)
            prepare_stroke_data(csv_path=source, output_dir=artifacts, split_seed=13,
                                balance_method="random_oversample")

            def prepare(*args, **kwargs):
                return prepare_stroke_data(csv_path=source, output_dir=artifacts, split_seed=13,
                                           balance_method="random_oversample")
            def metadata(*args, **kwargs):
                return load_stroke_metadata(artifacts / "metadata.json")

            common = patch.object(stroke_bnn_bp, "prepare_stroke_data", side_effect=prepare)
            with common, patch.object(stroke_bnn_bp, "load_stroke_metadata", side_effect=metadata), \
                 patch.object(stroke_bnn_bp, "get_device", return_value=torch.device("cpu")), \
                 patch.object(stroke_bnn_bp, "TRAIN_EPOCHS", 1), patch.object(stroke_bnn_bp, "BATCH_SIZE", 64):
                bp = stroke_bnn_bp.train_bnn_bp(101, split_seed=13, output_dir=output)
            with patch.object(stroke_bnn_dfa, "prepare_stroke_data", side_effect=prepare), \
                 patch.object(stroke_bnn_dfa, "load_stroke_metadata", side_effect=metadata), \
                 patch.object(stroke_bnn_dfa, "get_device", return_value=torch.device("cpu")), \
                 patch.object(stroke_bnn_dfa, "TRAIN_EPOCHS", 1), patch.object(stroke_bnn_dfa, "BATCH_SIZE", 64):
                dfa = stroke_bnn_dfa.train_dfa(103, split_seed=13, output_dir=output)
            rep_path = output / "synthetic-representation.pt"
            save_representation(dfa, rep_path, dfa["feature_names"])

            with patch("utils.stroke_representation.METADATA_PATH", artifacts / "metadata.json"), \
                 patch("utils.stroke_representation.RAW_PATH", source), \
                 patch("experiments.stroke_experiments.stroke_dfa_ste_binary.STE_EPOCHS", 2), \
                 patch("experiments.stroke_experiments.stroke_dfa_qubo.QUBO_NUM_READS", 2):
                head_results = [train_real_ls(rep_path), train_binarized_ls(rep_path),
                                train_direct_binary(rep_path, 107), train_ste_binary(rep_path, 107),
                                train_qubo(rep_path, 107), train_signed_least_squares(rep_path)]
            self.assertEqual(bp["model"].fc1.in_features, len(dfa["feature_names"]))
            self.assertEqual(len(dfa["y_train"]), dfa["H_train"].shape[0])
            self.assertEqual(metadata()["split_seed"], 13)
            self.assertTrue(all(np.isfinite(item["balanced_accuracy"]) for item in head_results))


if __name__ == "__main__":
    unittest.main()
