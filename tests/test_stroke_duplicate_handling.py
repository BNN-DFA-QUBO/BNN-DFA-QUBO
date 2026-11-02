"""Tests for Stroke duplicate run disambiguation and summary aggregation."""
import unittest
import pandas as pd
import numpy as np

from experiments.stroke_experiments.collect_stroke_results import deduplicate_latest_runs, build_stroke_summary


class TestStrokeDuplicateHandling(unittest.TestCase):
    def test_deduplicate_latest_runs_preserves_unique_and_disambiguates(self):
        # Synthetic DataFrame containing duplicate runs for seed 42 with different run_ids and timestamps
        data = [
            {"run_id": "run_old_42", "experiment": "stroke_baseline_ann", "method": "Standard ANN", "seed": 42, "balanced_accuracy": 0.725},
            {"run_id": "run_new_42", "experiment": "stroke_baseline_ann", "method": "Standard ANN", "seed": 42, "balanced_accuracy": 0.725},
            {"run_id": "run_123", "experiment": "stroke_baseline_ann", "method": "Standard ANN", "seed": 123, "balanced_accuracy": 0.750},
            {"run_id": "run_456", "experiment": "stroke_baseline_ann", "method": "Standard ANN", "seed": 456, "balanced_accuracy": 0.760},
            {"run_id": "run_789", "experiment": "stroke_baseline_ann", "method": "Standard ANN", "seed": 789, "balanced_accuracy": 0.770},
            {"run_id": "run_999", "experiment": "stroke_baseline_ann", "method": "Standard ANN", "seed": 999, "balanced_accuracy": 0.780},
        ]
        df = pd.DataFrame(data)
        self.assertEqual(len(df), 6)

        deduped, dups = deduplicate_latest_runs(df, ["experiment", "method", "seed"])
        # Exactly 2 duplicate rows identified for seed 42
        self.assertEqual(len(dups), 2)
        # Deduplicated dataset contains exactly 5 independent seed observations
        self.assertEqual(len(deduped), 5)
        # Verify it retained the latest run_id
        selected_42 = deduped[deduped["seed"] == 42].iloc[0]
        self.assertEqual(selected_42["run_id"], "run_new_42")

        # Check summary calculation
        summary = build_stroke_summary(deduped)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["runs"], 5)
        self.assertAlmostEqual(summary.iloc[0]["ba_mean"], np.mean([0.725, 0.750, 0.760, 0.770, 0.780]))


if __name__ == "__main__":
    unittest.main()

