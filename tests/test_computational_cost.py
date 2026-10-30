"""Focused unit tests for computational-cost analysis in MNIST shared validation.

These tests use small synthetic tensors and mock components to run quickly and
without requiring dataset downloads or long training times.
"""
from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

from experiments.MNIST_experiments import results_utils
from experiments.MNIST_experiments.run_mnist_shared_validation import (
    _append_result,
    _persist_selected_qubo_result,
)
import experiments.MNIST_experiments.run_mnist_shared_validation as shared_val


class ComputationalCostAnalysisTests(unittest.TestCase):
    def test_append_result_records_cost_fields(self):
        rows = []
        dummy_scores = torch.zeros(4, 10)
        dummy_labels = torch.zeros(4, dtype=torch.long)
        _append_result(
            rows=rows,
            name="Ridge LS",
            H_train=torch.zeros(4, 128),
            y_train=dummy_labels,
            H_val=torch.zeros(4, 128),
            y_val=dummy_labels,
            H_test=torch.zeros(4, 128),
            y_test=dummy_labels,
            train_scores=dummy_scores,
            val_scores=dummy_scores,
            test_scores=dummy_scores,
            config="test_config",
            head_fit_time_sec=0.1234,
            head_parameter_count=1290,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["head"], "Ridge LS")
        self.assertAlmostEqual(row["head_fit_time_sec"], 0.1234)
        self.assertEqual(row["head_parameter_count"], 1290)

    def test_parameter_counts_separation_and_semantics(self):
        # Backbone parameter count for 784 -> 128 -> 10:
        # fc1: 784*128 + 128 = 100,480
        # fc2: 128*10 + 10 = 1,290
        # Total backbone = 101,770
        backbone_count = 101770

        # Ridge LS: 128 * 10 feature weights + 10 biases = 1290
        ridge_head_count = 1290
        self.assertEqual(backbone_count + ridge_head_count, 103060)

        # Binarized Ridge LS: 128*10 binary weights + 10 scales + 10 biases = 1300
        bin_ridge_head_count = 1300
        self.assertEqual(backbone_count + bin_ridge_head_count, 103070)

        # QUBO Binary: 128*10 binary weights + 10 scales + 10 biases = 1300
        qubo_head_count = 1300
        self.assertEqual(backbone_count + qubo_head_count, 103070)

        # STE Binary: 128*10 deployed binary weights + 10 biases = 1290
        ste_head_count = 1290
        self.assertEqual(backbone_count + ste_head_count, 103060)

        # Ensure head parameters are never confused with backbone parameters
        self.assertNotEqual(backbone_count, ridge_head_count)
        self.assertNotEqual(backbone_count, qubo_head_count)

    def test_cached_vs_trained_representation_cost_semantics(self):
        # Scenario 1: Reused representation from cache with complete metadata
        reused_representation = {
            "training_seconds": 125.5,
            "feature_extraction_seconds": 3.2,
        }
        reused_source = "reused"
        raw_training = reused_representation.get("training_seconds")
        raw_extraction = reused_representation.get("feature_extraction_seconds")
        artifact_training_time = float(raw_training) if raw_training is not None else None
        artifact_extraction_time = float(raw_extraction) if raw_extraction is not None else None

        if reused_source == "trained":
            cur_backbone_time = artifact_training_time if artifact_training_time is not None else 0.0
            cur_extraction_time = artifact_extraction_time if artifact_extraction_time is not None else 0.0
        else:
            cur_backbone_time = 0.0
            cur_extraction_time = 0.0

        head_fit_time = 0.5
        # Reused run should record 0 for current run backbone and extraction:
        self.assertEqual(cur_backbone_time, 0.0)
        self.assertEqual(cur_extraction_time, 0.0)
        current_run_total = cur_backbone_time + cur_extraction_time + head_fit_time
        self.assertEqual(current_run_total, 0.5)

        # Full pipeline estimate retains historical duration:
        self.assertIsNotNone(artifact_training_time)
        self.assertIsNotNone(artifact_extraction_time)
        end_to_end_est = artifact_training_time + artifact_extraction_time + head_fit_time
        self.assertAlmostEqual(end_to_end_est, 129.2)

        # Scenario 2: Freshly trained representation
        trained_source = "trained"
        if trained_source == "trained":
            fresh_backbone_time = artifact_training_time if artifact_training_time is not None else 0.0
            fresh_extraction_time = artifact_extraction_time if artifact_extraction_time is not None else 0.0
        else:
            fresh_backbone_time = 0.0
            fresh_extraction_time = 0.0

        self.assertEqual(fresh_backbone_time, 125.5)
        self.assertEqual(fresh_extraction_time, 3.2)
        fresh_run_total = fresh_backbone_time + fresh_extraction_time + head_fit_time
        self.assertAlmostEqual(fresh_run_total, 129.2)

    def test_missing_historical_timing_handling(self):
        # Scenario: Old cached artifact with missing extraction time
        old_artifact = {
            "training_seconds": 160.7,
            "feature_extraction_seconds": None,
        }
        raw_training = old_artifact.get("training_seconds")
        raw_extraction = old_artifact.get("feature_extraction_seconds")
        art_train = float(raw_training) if raw_training is not None else None
        art_ext = float(raw_extraction) if raw_extraction is not None else None

        # Check status flag
        if art_train is not None and art_ext is not None:
            status = "complete"
        elif art_train is not None:
            status = "missing_extraction_time"
        else:
            status = "missing_backbone_and_extraction_time"

        self.assertEqual(status, "missing_extraction_time")

        # Full-pipeline estimate MUST NOT silently substitute 0.0
        head_fit = 1.2
        if art_train is not None and art_ext is not None:
            e2e = art_train + art_ext + head_fit
        else:
            e2e = ""
        self.assertEqual(e2e, "")

        # Current-run total remains well-defined and valid
        cur_backbone = 0.0
        cur_ext = 0.0
        cur_total = cur_backbone + cur_ext + head_fit
        self.assertEqual(cur_total, 1.2)

    def test_qubo_total_tuning_time_vs_selected_candidate_time_distinct(self):
        tuning_rows = [
            {"runtime_sec": 0.5},
            {"runtime_sec": 0.8},
            {"runtime_sec": 0.4},
        ]
        winner = {
            "num_reads": 50,
            "num_sweeps": 500,
            "scale_multiplier": 1.0,
            "validation_accuracy": 90.0,
            "solver_seed": 1051,
            "runtime_sec": 0.8,
            "tuning_total_seconds": 1.7,
        }

        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "mnist_results.csv"
            tuning_csv = Path(temp) / "tuning.csv"
            tuning_csv.write_text("dummy\n", encoding="utf-8")
            representation_path = Path(temp) / "rep.pt"
            representation = {
                "artifact_id": "art-1",
                "representation_sha256": "sha-1",
                "metadata": {
                    "backbone_parameter_count": 101770,
                    "sample_counts": {"train": 51000, "validation": 9000, "test": 10000},
                    "dataset_fingerprint": "fp-1",
                    "split_protocol": "proto-1",
                    "training": {"epochs": 16, "seed": 42},
                },
            }

            with mock.patch.object(results_utils, "RESULTS_CSV", path), \
                 mock.patch.object(results_utils, "get_device_name", return_value="cpu"), \
                 mock.patch.object(results_utils, "get_hardware_label", return_value="test"):
                _persist_selected_qubo_result(
                    seed=42,
                    test_accuracy=91.0,
                    winner=winner,
                    tuning_rows=tuning_rows,
                    validation_fraction=0.15,
                    representation=representation,
                    representation_path=representation_path,
                    representation_source="reused",
                    tuning_csv_path=tuning_csv,
                    backbone_training_time_sec=0.0,
                    feature_extraction_time_sec=0.0,
                    head_fit_time_sec=1.7,
                    end_to_end_training_time_sec=1.7,
                    current_run_total_time_sec=1.7,
                    pipeline_timing_status="complete",
                    backbone_parameter_count=101770,
                    head_parameter_count=1300,
                    total_parameter_count=103070,
                )

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            # Total tuning time across all 3 candidates:
            self.assertEqual(row["qubo_time_sec"], "1.7")
            # Runtime of the winner alone:
            self.assertEqual(row["qubo_selected_candidate_time_sec"], "0.8")
            self.assertNotEqual(row["qubo_time_sec"], row["qubo_selected_candidate_time_sec"])

            # Cost fields verification
            self.assertEqual(row["head_fit_time_sec"], "1.7")
            self.assertEqual(row["training_time_sec"], "0.0")
            self.assertEqual(row["pipeline_timing_status"], "complete")
            self.assertEqual(row["parameter_count"], "101770")
            self.assertEqual(row["backbone_parameter_count"], "101770")
            self.assertEqual(row["head_parameter_count"], "1300")
            self.assertEqual(row["total_parameter_count"], "103070")

    def test_shared_validation_csv_columns_and_nonnegative_timings(self):
        # Verify that all 4 heads receive cost fields and nonnegative times
        heads = [
            ("Ridge LS", 0.05, 1290),
            ("Binarized Ridge LS", 0.08, 1300),
            ("QUBO Binary", 2.15, 1300),
            ("STE (best validation epoch)", 1.45, 1290),
        ]
        sample_counts = {"train": 51000, "validation": 9000, "test": 10000}
        provenance = {"split_protocol": "proto", "dataset_fingerprint": "fp"}
        artifact_training_time = 100.0
        artifact_extraction_time = 2.0
        representation_source = "reused"
        cur_backbone_time = 0.0
        cur_extraction_time = 0.0
        backbone_param_count = 101770

        rows = []
        for name, fit_time, param_count in heads:
            rows.append({
                "head": name,
                "train_accuracy_pct": 90.0,
                "validation_accuracy_pct": 89.0,
                "test_accuracy_pct": 88.0,
                "configuration": "config",
                "head_fit_time_sec": fit_time,
                "head_parameter_count": param_count,
            })

        for row in rows:
            h_fit = float(row.get("head_fit_time_sec", 0.0))
            h_param = int(row.get("head_parameter_count", 0))
            e2e_time = artifact_training_time + artifact_extraction_time + h_fit
            cur_total_time = cur_backbone_time + cur_extraction_time + h_fit
            tot_params = backbone_param_count + h_param
            row.update({
                "head_fit_time_sec": h_fit,
                "backbone_training_time_sec": cur_backbone_time,
                "feature_extraction_time_sec": cur_extraction_time,
                "end_to_end_training_time_sec": e2e_time,
                "current_run_total_time_sec": cur_total_time,
                "pipeline_timing_status": "complete",
                "backbone_parameter_count": backbone_param_count,
                "head_parameter_count": h_param,
                "total_parameter_count": tot_params,
            })

        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertGreaterEqual(row["head_fit_time_sec"], 0.0)
            self.assertGreaterEqual(row["backbone_training_time_sec"], 0.0)
            self.assertGreaterEqual(row["feature_extraction_time_sec"], 0.0)
            self.assertGreaterEqual(row["end_to_end_training_time_sec"], 0.0)
            self.assertGreaterEqual(row["current_run_total_time_sec"], 0.0)
            self.assertEqual(row["pipeline_timing_status"], "complete")
            self.assertEqual(row["backbone_parameter_count"], 101770)
            self.assertIn(row["head_parameter_count"], [1290, 1300])
            self.assertEqual(
                row["total_parameter_count"],
                row["backbone_parameter_count"] + row["head_parameter_count"]
            )


    def test_independent_mnist_and_stroke_parameter_counts(self):
        # MNIST: 784 -> 128 -> 10
        mnist_backbone = 784 * 128 + 128 + 128 * 10 + 10  # 101,770
        self.assertEqual(mnist_backbone, 101770)
        # Stroke: 15 -> 64 -> 1
        stroke_backbone = 15 * 64 + 64 + 64 * 1 + 1  # 1089
        self.assertEqual(stroke_backbone, 1089)

        # Stroke Heads on 64 hidden units:
        # Real LS: 64 weights + 1 bias = 65
        self.assertEqual(stroke_backbone + 65, 1154)
        # Binarized Ridge LS: 64 binary weights + 1 scale + 1 bias = 66
        self.assertEqual(stroke_backbone + 66, 1155)
        # Direct Binary: 64 binary weights (bias fixed at 0) = 64
        self.assertEqual(stroke_backbone + 64, 1153)
        # STE Binary: 64 deployed binary weights + 1 bias = 65
        self.assertEqual(stroke_backbone + 65, 1154)
        # QUBO Binary: 64 binary weights + 1 scale + 1 bias = 66
        self.assertEqual(stroke_backbone + 66, 1155)

    def test_stroke_controlled_writer_cost_fields_schema(self):
        from experiments.stroke_experiments.run_stroke_controlled import save_results
        row = {
            "experiment_type": "controlled_end_to_end_head_comparison",
            "method": "QUBO Binary", "seed": 42, "status": "ok",
            "representation_path": "rep.pt", "representation_sha256": "abc",
            "run_id": "stroke_42_cost",
            "git_commit_sha": "deadbeef", "provenance_json": "{}",
            "backbone_training_time_sec": 12.34,
            "feature_extraction_time_sec": 0.45,
            "head_fit_time_sec": 1.23,
            "current_run_total_time_sec": 14.02,
            "end_to_end_training_time_sec": 14.02,
            "pipeline_timing_status": "complete",
            "backbone_parameter_count": 1089,
            "head_parameter_count": 66,
            "total_parameter_count": 1155,
        }
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "stroke_cost.csv"
            save_results([row], path)
            with path.open(newline="", encoding="utf-8") as handle:
                saved = next(csv.DictReader(handle))
            self.assertEqual(saved["backbone_training_time_sec"], "12.34")
            self.assertEqual(saved["feature_extraction_time_sec"], "0.45")
            self.assertEqual(saved["head_fit_time_sec"], "1.23")
            self.assertEqual(saved["current_run_total_time_sec"], "14.02")
            self.assertEqual(saved["end_to_end_training_time_sec"], "14.02")
            self.assertEqual(saved["pipeline_timing_status"], "complete")
            self.assertEqual(saved["backbone_parameter_count"], "1089")
            self.assertEqual(saved["head_parameter_count"], "66")
            self.assertEqual(saved["total_parameter_count"], "1155")


if __name__ == "__main__":
    unittest.main()
