# XAI implementation and sandbox re-audit report

**Audit date:** 4 November 2026
**Source baseline:** `BNN-DFA-QUBO/BNN-DFA-QUBO`, GitHub `main`, commit `f6ce345a61ae64ba7fb4ac7fcf2e73bc8978c157`
**Working tree:** clean-main-based project snapshot plus the new `xai/` implementation; Codex's unpushed local branch changes were not used.

## Executive result

The XAI code is integrated and the package-level/full available test suite is green in this sandbox. Four real Bank Marketing classifier-head explanation pilots completed. All 16 non-QUBO dataset/head combinations also completed end-to-end against controlled synthetic representations. CIFAR-10's real-data adapter uses the repository's current 256-dimensional representation and its non-QUBO heads pass synthetic integration tests.

**This is not yet a four-dataset real-data research validation.** Real MNIST, Stroke, and CIFAR-10 runs could not be executed because their data were not cached and this sandbox has no working network access. Actual QUBO solver runs and official-LIME runs could not be executed because `dwave-neal`/`dimod` and `lime` are unavailable and package downloads fail due DNS/network restrictions. Bank results use the project's explicitly recorded internal LIME-style weighted-Ridge fallback. The Bank pilots are single-seed, small explanation-sample pilots and are not publication-ready evidence.

## Repository baseline and integration scope

The source tree was checked against the live `main` tree at the commit above. It contains four established dataset tracks:

- MNIST: 10-class handwritten-digit classification.
- Stroke: binary stroke prediction, with a separate download/preprocessing step for the raw CSV.
- Bank Marketing: binary term-deposit prediction, with source data and preprocessed train/validation/test files in the repository.
- CIFAR-10: 10-class RGB image classification using the convolutional BNN-DFA model.

The separate XAI package supplies **20 hard-coded Python entry points**: five classifier-head methods for each of the four datasets (Real LS, Binarized LS, Direct Binary, STE Binary, QUBO). There is no CLI model-selector interface. The XAI layer uses the existing shared/frozen BNN-DFA representation pipeline and the repository's classifier-head procedures.

### Explicit explanation scope

The current XAI release explains the **classifier head over frozen BNN-DFA hidden representations**. Attribution features are latent units, not original pixels or raw Bank/Stroke columns. It currently does not claim to explain the separate end-to-end `baseline_ann`, `bnn_bp`, or native `bnn_dfa` runs; those require their own model-specific feature extraction and explanation protocol. CIFAR-10 covers its five shared-representation heads.

## Corrections made during this re-audit

1. **Continuous explanation target:** SHAP and LIME now explain the predicted-class-versus-strongest-competitor decision-score margin, with the competitor fixed for each explained row. This avoids probability saturation. `predicted_class_probability` remains separately logged for classification context; SHAP additivity is checked against the score margin, not probability.
2. **Consistent LIME output semantics:** the internal weighted-Ridge fallback now distinguishes per-SD coefficients from per-instance local contributions, matching the meaning of official LIME's `local_exp` coefficients and `local_pred` output. The backend used is saved in each manifest.
3. **Actual stability repetitions:** `stability_repeats` controls repeated SHAP runs with independent background samples and LIME runs with independent perturbation seeds. Stability summaries average finite correlations rather than silently relying on one repeat.
4. **Lazy optional QUBO dependency:** importing non-QUBO experiments and shared-representation utilities no longer requires `neal` to be importable at module import time. The solver is required only when QUBO optimization is invoked; the original Ising objective and solver calls were not replaced.
5. **MNIST STE integration:** the XAI STE head uses the same validation-selected 50-epoch STE training procedure without importing the shared runner's unrelated QUBO solver dependency.
6. **CIFAR-10 dimension reconciliation:** the current `CIFAR_BNN` and representation pipeline emit 256 hidden features. Old tests/comments/docs were stale at 128. Their current model-shape expectations were corrected; the model architecture was not changed. The dense feedback shape is `[10, 256]`, and the current model has 284,362 parameters.
7. **Test/runner reliability:** CIFAR representation test fixtures now supply the current required metadata and checkpoint-selection summary. The legacy MNIST QUBO result-persistence compatibility helper accepts its historical timing arguments and routes them through the current canonical persistence path. A manual MNIST data smoke script no longer tries downloading a dataset during pytest collection.
8. **Output hygiene:** obsolete early probability-target pilot folders were removed. Current Bank pilots are stored under `data/xai_results/bank/`; controlled synthetic integration outputs are under `data/xai_synthetic_validation/`.

No BNN/DFA/LS/QUBO training algorithm, model forward pass, architecture, or learned weight behavior was intentionally changed as part of the XAI integration. Integration changes are limited to optional-import timing and the test/documentation corrections described above.

## Automated tests and import checks

Executed from the project root in the supplied sandbox:

| Check | Result |
|---|---:|
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q -rs` | **152 passed, 5 skipped, 56 subtests passed** |
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q xai/tests` | **8 passed** |
| CIFAR-10 focused tests (model, DFA, data, heads, representations, runner) | **26 passed, 2 skipped** |
| `python -m compileall -q xai` and changed experiment modules | Passed |
| Import all 20 `xai.experiments.<dataset>.xai_<head>` entry points | **20/20 imported** |
| Four-dataset synthetic non-QUBO SHAP/LIME end-to-end matrix | **16/16 completed** |

The five full-suite skips are QUBO execution/integration tests that require `dwave-neal`, which is missing from this offline sandbox: CIFAR QUBO head, CIFAR QUBO-tuning runner, CIFAR five-head runner, and two Stroke all-head integration tests. QUBO validation tests that use mocked samplers did run and pass. No tests were marked passed merely by pretending a missing solver had executed.

The XAI tests include predictor-contract checks, finite/shape checks, SHAP local-accuracy checks, LIME fallback local-fit checks, decision-margin regression coverage, LIME coefficient/contribution semantics, stability/faithfulness metrics, and synthetic adapter predictions across all four dataset shapes and the four non-QUBO head procedures.

## Four-dataset synthetic integration matrix

A controlled synthetic harness exercised the complete XAI runner for all four non-QUBO heads (Real LS, Binarized LS, Direct Binary, STE) on each dataset's expected hidden-feature/class shape:

| Dataset adapter | Feature width | Classes | End-to-end methods |
|---|---:|---:|---:|
| MNIST | 128 | 10 | 4/4 |
| Stroke | 64 | 2 | 4/4 |
| Bank | 64 | 2 | 4/4 |
| CIFAR-10 | 256 | 10 | 4/4 |

All **16/16** synthetic paths produced finite output files, SHAP additivity pass rate 1.0, and finite LIME metrics. LIME local weighted R² ranged from approximately 0.99970 to 0.99993 on the synthetic linear head fixtures. SHAP additivity residuals were below `2e-13`. Synthetic attribution rank agreement varied across generated random fixtures and is not treated as a scientific quality result. These synthetic runs demonstrate plumbing and numerical contracts only; they are not real-data experiment results.

The four QUBO experiment entry points import. Mocked QUBO solver protocol/unit tests pass, but actual QUBO annealing was not run because the solver package is unavailable.

## Real Bank Marketing pilot experiments

Bank is the only dataset with its real data available in this sandbox. Four actual XAI head experiments completed using the existing Bank train/validation/test data and one shared frozen BNN-DFA representation:

- Train: 28,831 rows; validation: 6,179 rows; test: 6,178 rows.
- Seed: 42.
- Representation: 15-epoch BNN-DFA artifact, 64 latent features.
- Shared representation SHA-256: `d8595b0ee65f6d6093d1a6327ef31aa84d52838f38a2e2c58c256d46582c269b`.
- Each head's classifier was evaluated on the entire 6,178-row held-out test split. Eight held-out test rows per run were selected deterministically for explanations; this small subset is not the basis of the classification metrics.
- Background size: 40 training examples; SHAP `nsamples`: 2,176; LIME-style perturbations: 2,000; stability repetitions: 2.
- Explanation target: predicted-class-versus-strongest-competitor score margin.

| Head | Test accuracy | Balanced accuracy | Macro-F1 | Explained rows | Mean LIME local R² | SHAP/LIME rank Spearman | SHAP stability Spearman | LIME stability Spearman | SHAP top-k/random ablation ratio | LIME top-k/random ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Real LS | 0.810942 | 0.852075 | 0.700649 | 8 | 0.999998 | 0.998565 | 0.998510 | 0.999996 | 3.492 | 3.664 |
| Binarized LS | 0.701845 | 0.778058 | 0.600120 | 8 | 0.999998 | 0.996960 | 0.997356 | 0.999996 | 3.214 | 3.831 |
| Direct Binary | 0.672871 | 0.741035 | 0.571668 | 8 | 0.999998 | 0.996960 | 0.997356 | 1.000000 | 3.417 | 3.779 |
| STE Binary | 0.707349 | 0.786804 | 0.606117 | 8 | 0.999998 | 0.996938 | 0.997356 | 0.999993 | 3.042 | 3.190 |

Across these runs, every SHAP additivity check passed. The largest per-row absolute additivity residual was below `2e-15`. Attribution arrays were finite, and no explainer warnings were captured. The top-k/random ratios are **perturbation sensitivity diagnostics** relative to training-median replacement, not causal effects or proof of global faithfulness. Very high LIME local R² is expected for these linear classifier-head decision functions and should not be interpreted as general model-agnostic explanatory validity.

Every saved Bank run has `feature_attributions.csv`, `sample_metrics.csv`, `summary_metrics.csv`, `run_manifest.json`, a per-run `VALIDATION_REPORT.md`, and SHAP/LIME comparison plots. Manifests include the feature scope, exact representation SHA, split counts, background size, explanation sample indices, model metrics, output backend, and checks.

## What could not be run in this sandbox

- **Real MNIST:** the dataset was not present in the cache. A download attempt failed because DNS/network is disabled.
- **Real CIFAR-10:** not cached; torchvision download could not run offline. The current 256-feature adapter and synthetic explainer path passed, but there is no real-data CIFAR result in this archive.
- **Real Stroke:** the raw Stroke CSV was not in the source tree or sandbox cache. The repo's `experiments/stroke_experiments/download_stroke.py` prerequisite needs to be run in a network-enabled environment.
- **Actual QUBO solver:** `dwave-neal`/`neal` was missing. An attempt to install `dimod`, `dwave-neal`, and `lime` failed due network/DNS restrictions. QUBO code paths and solver protocol tests are covered by imports/mocks, but no annealing result is claimed.
- **Official LIME:** `lime` was missing and could not be installed. The Bank runs use the manifest-recorded `internal_lime_weighted_ridge` fallback. Install `xai/requirements.txt`, rerun, and verify the manifest reports `lime.lime_tabular.LimeTabularExplainer` before calling the output official-LIME research data.
- **GPU:** this sandbox is CPU-only (`torch 2.10.0+cpu`). This does not change the source model but is not a performance validation of the user's RTX GPU environment.

## Research-readiness judgement

The **code integration and executable non-QUBO XAI paths pass the available checks**, and the real Bank pilots show consistent numerical behavior on one seed. The bundle is appropriate as an integrated implementation and pilot-validation package, but it is **not yet a publication-ready four-dataset results package**. Before paper claims, run multi-seed experiments on all four real datasets, execute the real QUBO solver, install and use official LIME, prespecify explanation sample/background/SHAP/LIME budgets, report across-seed uncertainty, and retain raw manifests/results. Keep latent-feature explanations distinct from original-input explanations.
