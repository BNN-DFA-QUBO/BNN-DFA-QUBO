"""Regression tests for CIFAR-10 QUBO validation tuning diagnostics and CSV logging."""
import csv
import os
from pathlib import Path
import pytest
import torch

from experiments.CIFAR10_experiments.tune_qubo_validation import (
    DEFAULT_NUM_READS,
    DEFAULT_NUM_SWEEPS,
    DEFAULT_SCALE_MULTIPLIERS,
    select_configuration,
    tune_from_representations,
    validate_grid,
    write_tuning_csv,
)
from tests.test_cifar10_data import MockCIFAR10Dataset
from utils.CIFAR10.representation import (
    build_metadata,
    representation_path,
    save_representation,
)
from experiments.CIFAR10_experiments.run_cifar10_shared_validation import (
    run_cifar10_shared_validation,
)


def _sample_rows():
    return [
        {
            "num_reads": 25,
            "scale_multiplier": 0.5,
            "num_sweeps": 250,
            "solver_seed": 1001,
            "validation_accuracy": 31.5,
            "selected": False,
            "status": "ok",
            "error": "",
            "runtime_sec": 1.25,
            "tuning_total_seconds": 5.50,
            "selected_candidate_runtime_sec": "",
        },
        {
            "num_reads": 50,
            "scale_multiplier": 1.0,
            "num_sweeps": 500,
            "solver_seed": 1002,
            "validation_accuracy": 34.2,
            "selected": True,
            "status": "ok",
            "error": "",
            "runtime_sec": 2.10,
            "tuning_total_seconds": 5.50,
            "selected_candidate_runtime_sec": 2.10,
        },
        {
            "num_reads": 100,
            "scale_multiplier": 2.0,
            "num_sweeps": 1000,
            "solver_seed": 1003,
            "validation_accuracy": 0.0,
            "selected": False,
            "status": "failed",
            "error": "RuntimeError: simulated annealing timeout",
            "runtime_sec": 2.15,
            "tuning_total_seconds": 5.50,
            "selected_candidate_runtime_sec": "",
        },
    ]


def test_write_tuning_csv_creates_file(tmp_path):
    """Test that write_tuning_csv creates an isolated non-empty CSV file."""
    rows = _sample_rows()
    csv_path = write_tuning_csv(
        rows=rows,
        seed=42,
        epochs=20,
        validation_fraction=0.1,
        output_dir=tmp_path,
        representation_artifact_id="art_test_123",
        n_train=45000,
        n_validation=5000,
        n_test=10000,
    )

    assert csv_path.exists()
    assert csv_path.is_file()
    assert csv_path.parent == tmp_path
    assert "cifar10_qubo_tuning_seed_42_epochs_20_val_0p1000_art_art_test_123" in csv_path.name
    assert csv_path.stat().st_size > 0


def test_write_tuning_csv_expected_columns(tmp_path):
    """Test that the CSV output contains all required diagnostic columns and NO test_accuracy."""
    rows = _sample_rows()
    csv_path = write_tuning_csv(
        rows=rows,
        seed=42,
        epochs=20,
        validation_fraction=0.1,
        output_dir=tmp_path,
        representation_artifact_id="art_test_123",
        n_train=45000,
        n_validation=5000,
        n_test=10000,
    )

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

    expected_cols = [
        "seed",
        "epochs",
        "validation_fraction",
        "representation_artifact_id",
        "n_train",
        "n_validation",
        "n_test",
        "num_reads",
        "num_sweeps",
        "scale_multiplier",
        "solver_seed",
        "validation_accuracy",
        "selected",
        "runtime_sec",
        "selected_candidate_runtime_sec",
        "tuning_total_seconds",
        "status",
        "error",
    ]
    assert header == expected_cols

    # Critical requirement: strictly exclude test accuracy from candidate tuning/selection
    assert "test_accuracy" not in header
    assert "test_acc" not in header


def test_write_tuning_csv_candidate_rows_and_metadata(tmp_path):
    """Test that candidate values and experiment metadata are accurately recorded in every row."""
    rows = _sample_rows()
    csv_path = write_tuning_csv(
        rows=rows,
        seed=123,
        epochs=15,
        validation_fraction=0.15,
        output_dir=tmp_path,
        representation_artifact_id="art_cifar10_abc",
        n_train=42500,
        n_validation=7500,
        n_test=10000,
    )

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    assert len(records) == len(rows)

    for i, rec in enumerate(records):
        # Metadata check on every row
        assert int(rec["seed"]) == 123
        assert int(rec["epochs"]) == 15
        assert float(rec["validation_fraction"]) == pytest.approx(0.15)
        assert rec["representation_artifact_id"] == "art_cifar10_abc"
        assert int(rec["n_train"]) == 42500
        assert int(rec["n_validation"]) == 7500
        assert int(rec["n_test"]) == 10000

        # Candidate specific check
        expected = rows[i]
        assert int(rec["num_reads"]) == expected["num_reads"]
        assert int(rec["num_sweeps"]) == expected["num_sweeps"]
        assert float(rec["scale_multiplier"]) == pytest.approx(expected["scale_multiplier"])
        assert int(rec["solver_seed"]) == expected["solver_seed"]
        assert float(rec["validation_accuracy"]) == pytest.approx(expected["validation_accuracy"])
        assert rec["status"] == expected["status"]
        assert rec["error"] == expected["error"]


def test_selected_candidate_identification(tmp_path):
    """Test that the selected candidate status is accurately flagged in the tuning CSV."""
    rows = _sample_rows()
    # Winner selection function
    successful = [r for r in rows if r["status"] == "ok"]
    winner = select_configuration(successful)
    assert winner["num_reads"] == 50
    assert winner["scale_multiplier"] == 1.0

    csv_path = write_tuning_csv(
        rows=rows,
        seed=42,
        epochs=20,
        validation_fraction=0.1,
        output_dir=tmp_path,
    )

    with open(csv_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    # Exactly one candidate should be selected
    selected_flags = [rec["selected"] for rec in records]
    assert selected_flags == ["False", "True", "False"]

    selected_row = [rec for rec in records if rec["selected"] == "True"][0]
    assert int(selected_row["num_reads"]) == 50
    assert float(selected_row["validation_accuracy"]) == pytest.approx(34.2)


def test_no_test_set_leakage_in_configuration_selection():
    """Verify that configuration selection is determined strictly by validation accuracy.

    Manipulating test data/labels must have zero effect on candidate selection.
    """
    rows = [
        {"num_reads": 25, "num_sweeps": 250, "scale_multiplier": 1.0, "validation_accuracy": 30.0},
        {"num_reads": 50, "num_sweeps": 500, "scale_multiplier": 1.0, "validation_accuracy": 35.0},
        {"num_reads": 100, "num_sweeps": 1000, "scale_multiplier": 1.0, "validation_accuracy": 32.0},
    ]

    selected = select_configuration(rows)
    assert selected["num_reads"] == 50
    assert selected["validation_accuracy"] == 35.0

    # Even if an external 'test_accuracy' was maliciously injected into the rows dict,
    # select_configuration must ignore it completely and base choice strictly on validation_accuracy.
    rows_with_fake_test = [
        {"num_reads": 25, "num_sweeps": 250, "scale_multiplier": 1.0, "validation_accuracy": 30.0, "test_accuracy": 99.9},
        {"num_reads": 50, "num_sweeps": 500, "scale_multiplier": 1.0, "validation_accuracy": 35.0, "test_accuracy": 10.0},
        {"num_reads": 100, "num_sweeps": 1000, "scale_multiplier": 1.0, "validation_accuracy": 32.0, "test_accuracy": 80.0},
    ]
    selected_after = select_configuration(rows_with_fake_test)
    assert selected_after["num_reads"] == 50
    assert selected_after["validation_accuracy"] == 35.0


def test_shared_validation_runner_writes_tuning_csv_when_enabled(tmp_path, monkeypatch):
    """Integration test: verify run_cifar10_shared_validation writes tuning CSV when fixed_qubo=False."""
    # Direct tuning CSV output to isolated tmp_path
    monkeypatch.setenv("CIFAR10_QUBO_TUNING_DIR", str(tmp_path))
    results_csv = tmp_path / "cifar10_results.csv"
    monkeypatch.setenv("CIFAR10_RESULTS_CSV", str(results_csv))

    seed = 42
    epochs = 1
    val_frac = 0.2
    train_ds = MockCIFAR10Dataset(num_samples=60)
    test_ds = MockCIFAR10Dataset(num_samples=20)
    rep_dir = tmp_path / "reps"

    metadata = build_metadata(
        train_ds, test_ds, seed=seed, epochs=epochs, validation_fraction=val_frac
    )
    path = representation_path(rep_dir, seed=seed, epochs=epochs, validation_fraction=val_frac)
    indices = metadata["split_indices"]

    data = {
        "H_train": torch.randn(len(indices["train"]), 128),
        "y_train": torch.as_tensor([train_ds.targets[i] for i in indices["train"]], dtype=torch.long),
        "H_validation": torch.randn(len(indices["validation"]), 128),
        "y_validation": torch.as_tensor([train_ds.targets[i] for i in indices["validation"]], dtype=torch.long),
        "H_test": torch.randn(len(indices["test"]), 128),
        "y_test": torch.as_tensor([test_ds.targets[i] for i in indices["test"]], dtype=torch.long),
        "training_seconds": 0.1,
        "feature_extraction_seconds": 0.1,
    }
    save_representation(path, data, metadata)

    # Run with small custom grid for speed
    results = run_cifar10_shared_validation(
        seed=seed,
        epochs=epochs,
        ste_epochs=2,
        direct_steps=5,
        validation_fraction=val_frac,
        representation_dir=rep_dir,
        fixed_qubo=False,
        num_reads_grid=(2,),
        num_sweeps_grid=(5,),
        scale_multipliers=(1.0,),
        device="cpu",
        train_dataset=train_ds,
        test_dataset=test_ds,
    )
    assert len(results) == 5

    # Check that tuning CSV was written to tmp_path
    tuning_csvs = list(tmp_path.glob("cifar10_qubo_tuning_*.csv"))
    assert len(tuning_csvs) == 1, f"Expected 1 tuning CSV in tmp_path, found: {tuning_csvs}"
    csv_file = tuning_csvs[0]

    with open(csv_file, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    assert len(records) == 1  # 1 candidate in custom grid
    rec = records[0]
    assert int(rec["seed"]) == seed
    assert int(rec["epochs"]) == epochs
    assert rec["selected"] == "True"
    assert rec["status"] == "ok"
    assert int(rec["num_reads"]) == 2
    assert int(rec["num_sweeps"]) == 5


def test_write_tuning_csv_empty_rows(tmp_path):
    """Test that write_tuning_csv handles empty rows gracefully, writing header only."""
    csv_path = write_tuning_csv(
        rows=[],
        seed=42,
        epochs=10,
        validation_fraction=0.1,
        output_dir=tmp_path,
        representation_artifact_id="art_empty",
    )
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 1  # Header only
    assert "validation_accuracy" in reader[0]


def test_write_tuning_csv_failed_candidate_preserved(tmp_path):
    """Test that candidate failures and specific error messages are captured in the CSV."""
    rows = [
        {
            "num_reads": 25,
            "scale_multiplier": 1.0,
            "num_sweeps": 250,
            "solver_seed": 777,
            "validation_accuracy": 0.0,
            "selected": False,
            "status": "failed",
            "error": "MemoryError: simulated annealing out of memory",
            "runtime_sec": 0.5,
            "tuning_total_seconds": 0.5,
        }
    ]
    csv_path = write_tuning_csv(
        rows=rows,
        seed=777,
        epochs=5,
        validation_fraction=0.1,
        output_dir=tmp_path,
    )
    with open(csv_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert "MemoryError: simulated annealing out of memory" in records[0]["error"]
    assert records[0]["selected"] == "False"
