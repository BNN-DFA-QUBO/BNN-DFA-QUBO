# MNIST + Stroke Remediation Acceptance Report

Scope: bounded code and synthetic-test remediation only. No model training, data download, multi-seed experiment, or QUBO sweep was run. Final multi-seed study outputs are not present and remain unverified.

| Checklist item | Status | Evidence and limitation |
|---|---|---|
| Runner and comparison protocol separation | PASS | README and runner metadata distinguish MNIST independent backbones from shared frozen-head comparison, and Stroke controlled end-to-end from fixed-representation head-seed variability. MNIST representation tests verify split-index label alignment; Stroke v3 validates row IDs and stored labels. A shared seed alone is not used as evidence of identical representations. |
| Data split and preprocessing leakage controls | PASS | MNIST representation metadata records official train stratified train/validation indices and official test indices; representation loading verifies dataset fingerprint and label ordering. Stroke preprocessing tests cover partition metadata, train-only transforms/balancing, and split artifact compatibility. No source dataset was downloaded or regenerated. |
| LS, binarized LS, Direct Binary, and STE head invariants | PASS | Existing Ridge/binarized tests plus `tests.test_direct_binary_head`, `test_model_invariants`, and `test_ste_head` cover synthetic dimensions, sign conventions, finite scores, and STE gradient behavior. These tests do not establish comparative model performance. |
| Validation-only tuning and locked test evaluation | PASS | MNIST tuner receives only train/validation arrays for candidate fit/rank; runner computes selected test scores after winner selection. Stroke QUBO/STE candidate paths rank on validation data and apply the selected threshold/configuration to test afterward. The MNIST read sweep is separately marked descriptive. |
| QUBO objective, spin mapping, and solver boundary | PASS | `models/qubo_head.py` and Stroke QUBO tests compare small-instance direct objective/Ising energies and validate solver settings/responses; `test_model_invariants` measures the existing coupling cutoff tolerance. No objective or cutoff change was made in this remediation. |
| MNIST QUBO timing semantics | FIXED, TESTED | `tune_qubo_validation.tune_from_representations` records full tuner duration separately from winner candidate duration; shared result paths persist both. Candidate status/error/runtime is recorded. Historical seed-123 timing remains untouched and is not valid efficiency evidence. |
| Failed MNIST candidate artifacts | FIXED, TESTED | Candidate CSV and failure JSON are saved for standalone all-candidate failures; shared validation saves candidate diagnostics and stops before test evaluation when all candidates fail. |
| MNIST test sweep interpretation | FIXED, TESTED | `sweep_qubo_num_reads_shared.py` marks test fields `post_hoc_descriptive_test_metric_not_for_selection`; its existing semantics and grid are unchanged. |
| MNIST representation validation and split isolation | PASS | `utils/MNIST/representation.py` and shared runner bind dataset/split metadata, provenance and tensor checksums; selection API receives train and validation only and scores test after selection. No new dataset run was performed. |
| Stroke QUBO fields in fixed-representation multiseed output | FIXED, TESTED | `qubo_result_columns` preserves total/phase timings, reads, sweeps, alpha settings, candidates, solver seeds, validation metrics, and threshold; summary groups distinguish representation/configuration. |
| Stroke representation tensor integrity | FIXED, TESTED | `utils/stroke_representation.py` v3 hashes features, labels and metadata; load validates shape, finite values, counts, split identities, and label-to-row mapping. Tests cover corruption and mismatch. |
| Legacy Stroke representations | PASS | v2 artifacts load with `legacy_unverified` warning and are not reported as v3 verified. They lack content hashes/exact row-label proof. |
| Stroke representation compatibility at head loaders | FIXED, TESTED | All standalone Stroke classifier loaders use `load_stroke_head_representation`, which checks the project hidden dimension and current preprocessing/dataset metadata when available. The fixed-representation multiseed runner does not compare its head seeds with the backbone seed. `validate_stroke_ls_heads` supplies its known backbone seed, dimensions, and training configuration. `load_representation` warns and continues when optional expected metadata is absent; legacy v2 remains `legacy_unverified`. Standalone head CLIs do not request an epoch/training config, so none is invented or compared there. |
| Provenance and stable run identifiers | FIXED, TESTED | New shared result/controlled/multiseed rows carry run IDs and provenance. Dirty worktree fingerprint includes tracked diffs and untracked file contents; unavailable values remain labeled unavailable. Package/device metadata is configuration provenance, not a guarantee of bitwise reproducibility. |
| Duplicate protection and output preservation | FIXED, TESTED | MNIST append checks run IDs; controlled Stroke and multiseed writers reject duplicate IDs and refuse existing outputs absent explicit overwrite. Historical CSV columns/rows are covered by schema tests. |
| Aggregate protocol integrity | FIXED, TESTED | MNIST aggregation groups by protocol and selected QUBO settings; Stroke multiseed summaries distinguish representation hashes and QUBO settings. Aggregate generation was tested on synthetic records only. |
| Direct Binary synthetic coverage | FIXED, TESTED | `tests/test_direct_binary_head.py` covers dimensions, bipolar signs, finite output, and determinism on synthetic inputs. |
| STE forward/backward coverage | PASS — existing tests | `tests.test_model_invariants` and `experiments.MNIST_experiments.test_ste_head` cover bipolar forward values and surrogate gradient behavior; no dataset training was run. |
| QUBO energy and solver validation | PASS — targeted tests | MNIST and Stroke tests cover small-instance energy equivalence, cutoff tolerance, invalid settings, malformed responses, non-finite/overflow coefficients, and contextual solver failures. Numerical tolerance is reported by the test output. |
| Five-seed MNIST/Stroke outcomes and aggregate regeneration | NOT VERIFIED | Final current-protocol raw multi-seed runs are absent. The seed-42 pilot is not conclusive; historical Stroke v2 rows are not current v3 evidence. Do not regenerate paper tables until compatible raw runs exist. |
| Cross-platform bitwise reproducibility | NOT APPLICABLE | The repository records configuration/provenance but does not demonstrate identical floating-point outputs across devices/platforms. |

## Verification executed

All commands below ran against the final working tree and passed:

| Command | Result |
|---|---|
| `\.venv\Scripts\python.exe -m unittest tests.test_model_invariants tests.test_qubo_validation_tuning` | 26 tests, PASS |
| `\.venv\Scripts\python.exe -m unittest tests.test_binarized_ridge_audit tests.test_mnist_representation tests.test_mnist_results_schema tests.test_model_invariants tests.test_remediation_integrity` | 22 tests, PASS |
| `\.venv\Scripts\python.exe -m unittest experiments.MNIST_experiments.test_ste_head tests.test_direct_binary_head` | 3 tests, PASS |
| `\.venv\Scripts\python.exe -m unittest experiments.stroke_experiments.test_stroke_qubo` | 9 tests, PASS |
| `\.venv\Scripts\python.exe -m unittest experiments.stroke_experiments.test_stroke_preprocessing` | 24 tests, PASS |
| `\.venv\Scripts\python.exe -m unittest experiments.stroke_experiments.test_stroke_ls_heads` | 7 tests, PASS |
| `\.venv\Scripts\python.exe -m py_compile` on all changed Python files | PASS |
| `git diff --check` | PASS |

## Historical MNIST seed-123 note

The existing row in `data/results/mnist_results.csv` was inspected and left unchanged. It records `qubo_time_sec=6.899997970322147e-06` and has no candidate-time field. The associated older tuning CSV has no per-candidate `runtime_sec`; do not use that row for efficiency claims or infer replacement timing.

## Before final multi-seed studies

1. Resolve the requested Stroke representation training-configuration contract for workflows that reuse external representations.
2. Run the current fixed-split multi-seed protocols into fresh output directories, retaining raw rows and representation artifacts.
3. Regenerate aggregates only from raw records sharing dataset, protocol, split/preprocessing identity, and selected configuration.
4. Treat fixed-split seed variability separately from split-to-split generalization and do not use the MNIST descriptive read sweep to choose parameters.
