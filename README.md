# BNN-DFA-QUBO

Research code for comparing Binarized Neural Networks (BNNs), Direct Feedback Alignment (DFA), and classifier heads, including least-squares (LS), binary, straight-through estimator (STE), and QUBO/Ising heads.

This repository contains **two distinct experiment tracks**:

- **MNIST:** ten-class digit classification with BNN backpropagation, DFA, LS-derived heads, STE, and QUBO.
- **Stroke:** binary stroke prediction with saved preprocessing metadata, stratified train/validation/test splits, optional training-only class balancing, and five DFA classifier heads.

Run commands from the repository root. Generated data, representations, and experiment outputs are local files and are not committed by default.

## 1. Set up the environment

Use a Python interpreter compatible with the pinned PyTorch and torchvision versions in `requirements.txt`. A virtual environment is recommended.

### Windows PowerShell

```powershell
git clone https://github.com/BNN-DFA-QUBO/BNN-DFA-QUBO.git
cd BNN-DFA-QUBO

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If PowerShell blocks virtual-environment activation, use the activation guidance from your local Python installation or activate the environment in Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
git clone https://github.com/BNN-DFA-QUBO/BNN-DFA-QUBO.git
cd BNN-DFA-QUBO

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Verify the installation

```powershell
python -c "import torch, torchvision, numpy, pandas; import models, utils; print('Imports OK'); print('PyTorch:', torch.__version__)"
python -m experiments.MNIST_experiments.test_data
python -m unittest experiments.stroke_experiments.test_stroke_preprocessing
python -m unittest experiments.stroke_experiments.test_stroke_ls_heads
```

The MNIST data loader downloads MNIST the first time it is used. The Stroke pipeline downloads its CSV in the separate step described below.

## 2. MNIST experiments

### 2.1 Overview of the Paper's MNIST Experiments

The paper (*"Improving Direct Feedback Alignment in Binarized Neural Networks using QUBO Optimization of the Binary Classifier Head"*) evaluates eight distinct methods on the 10-class MNIST digit dataset (784 -> 128 -> 10):
1. **Three end-to-end baselines:** Standard full-precision ANN, BNN trained with Backpropagation (BP), and native BNN trained with Direct Feedback Alignment (DFA).
2. **Five classifier head procedures:** Real Least Squares (Real LS), Binarized Least Squares (Binarized LS), Direct Binary, Straight-Through Estimator Binary (STE Binary), and QUBO Binary.

> [!IMPORTANT]
> **Independent Backbones vs. Shared Representations:**
> In the paper's primary MNIST protocol (Section III-A, IV-B, and Tables III & IV), **each classifier head script independently trains its own BNN-DFA backbone** from scratch on all 60,000 MNIST training images. Therefore, running individual head scripts does **not** evaluate heads on a shared, frozen representation and does not isolate the effect of head optimization alone.
>
> To isolate classifier head performance on an identical feature extractor, the repository also implements a **shared-representation validation protocol** (`run_mnist_shared_validation.py`), which trains one BNN-DFA backbone per seed, splits the training set into 51,000 train and 9,000 validation images, freezes the hidden features, and compares heads under identical representations. These two protocols must not be confused or silently combined.

---

### 2.2 The Eight Individual MNIST Methods

The table below maps all eight methods reported in the paper to their exact implementation scripts in `experiments/MNIST_experiments/`, verified PowerShell commands, purposes, output locations, and backbone training modes:

| Method in Paper | Exact Python Script | Exact Verified PowerShell Command | Purpose & Description | Main Outputs & Locations | Backbone Training Mode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard ANN** | `experiments/MNIST_experiments/baseline_ann.py` | `.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.baseline_ann --seed 42` | End-to-end full-precision feedforward baseline (784 -> 128 -> 10) trained with backpropagation and cross-entropy loss. | Appends run metrics (accuracy, loss, training time, parameter count) to `data/results/mnist_results.csv` (or `$env:BNN_DFA_RESULTS_CSV`). | Trains its own ANN network |
| **BNN with Backpropagation** | `experiments/MNIST_experiments/bnn_bp.py` | `.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.bnn_bp --seed 42` | End-to-end binary neural network (784 -> 128 -> 10) with scaled binary forward weights {-alpha, +alpha} and STE backpropagation. | Appends run metrics to `data/results/mnist_results.csv` (or `$env:BNN_DFA_RESULTS_CSV`). | Trains its own BNN network via BP |
| **BNN with Direct Feedback Alignment (DFA)** | `experiments/MNIST_experiments/bnn_dfa.py` | `.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.bnn_dfa --seed 42` | End-to-end BNN trained with Direct Feedback Alignment using a fixed random projection matrix B in R^(10x128) instead of transpose weights. | Appends run metrics to `data/results/mnist_results.csv` (or `$env:BNN_DFA_RESULTS_CSV`). | Trains its own BNN network via DFA |
| **DFA + Real-Valued Least-Squares Head** | `experiments/MNIST_experiments/dfa_least_squares.py` | `.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.dfa_least_squares --seed 42 --ls-solver legacy` *(paper)*<br>`.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.dfa_least_squares --seed 42` *(Ridge default)* | Trains a BNN-DFA backbone, extracts 128-dim ReLU hidden features H in R^(60000x128), augments with bias column [H, 1], and fits continuous weights to one-hot {0, 1} targets. | Appends run metrics to `data/results/mnist_results.csv` (or `$env:BNN_DFA_RESULTS_CSV`). | Retrains its own BNN-DFA backbone |
| **DFA + Binarized Least-Squares Head** | `experiments/MNIST_experiments/dfa_binary_head.py` | `.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.dfa_binary_head --seed 42 --ls-solver legacy` *(paper)*<br>`.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.dfa_binary_head --seed 42` *(Ridge default)* | Fits an initial continuous least-squares head on its trained BNN-DFA features, takes weight signs sign(W), and fits a class-specific positive scale through the origin while retaining the real LS bias. | Appends run metrics to `data/results/mnist_results.csv` (or `$env:BNN_DFA_RESULTS_CSV`). | Retrains its own BNN-DFA backbone |
| **DFA + Direct Binary Head** | `experiments/MNIST_experiments/dfa_direct_binary_head.py` | `.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.dfa_direct_binary_head --seed 42` | Trains a tanh-relaxed latent weight parameterization W_bin = sign(tanh(W_latent)) on frozen BNN-DFA features for 200 steps with Adam, cross-entropy, and learnable bias. | Appends run metrics to `data/results/mnist_results.csv` (or `$env:BNN_DFA_RESULTS_CSV`). | Retrains its own BNN-DFA backbone |
| **DFA + STE Binary Head** | `experiments/MNIST_experiments/dfa_binary_head_ste.py` | `.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.dfa_binary_head_ste --seed 42` | Fits a binary classifier head on frozen BNN-DFA features using sign binarization and straight-through estimator gradients for 50 full-batch Adam steps with cross-entropy and learned bias (no scale parameter). | Appends run metrics to `data/results/mnist_results.csv` (or `$env:BNN_DFA_RESULTS_CSV`). | Retrains its own BNN-DFA backbone |
| **BNN-DFA + QUBO Binary Head** | `experiments/MNIST_experiments/bnn_dfa_qubo.py` | `.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.bnn_dfa_qubo --seed 42 --qubo-scale-solver legacy --num-reads 100` *(paper)*<br>`.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.bnn_dfa_qubo --seed 42` *(Ridge default)* | Formulates one-vs-rest binary classifier optimization as an Ising/QUBO problem over 128 spins per class with {-1, +1} targets. Solves using classical simulated annealing (dwave-neal, 100 reads), then fits continuous scale and bias. | Appends run metrics (including qubo_time_sec) to `data/results/mnist_results.csv` (or `$env:BNN_DFA_RESULTS_CSV`). | Retrains its own BNN-DFA backbone |

> [!NOTE]
> **Solver Modes (`--ls-solver` / `--qubo-scale-solver`):**
> - `--ls-solver legacy`: Reproduces the paper's original float32 unregularized `torch.linalg.lstsq` solve.
> - `--ls-solver ridge` *(default)*: Uses the numerically stabilized float64 SVD Ridge solver with unregularized intercept (`lambda_relative=1e-4`), preventing ill-conditioned inversions.
> - `--qubo-scale-solver legacy`: Uses the unregularized float32 least-squares fit to initialize the base class scales alpha_k.
> - `--qubo-scale-solver ridge` *(default)*: Uses the stabilized Ridge solve to initialize alpha_k.

---

### 2.3 Experiment Runners and Diagnostic Scripts

The repository provides several experiment runners and validation scripts in `experiments/MNIST_experiments/`:

#### 1. Multi-Seed Individual Runner: `run_mnist_controlled.py`
- **Purpose:** Executes individual MNIST experiments across multiple seeds sequentially. Each experiment script retrains its own network and prints its test accuracy.
- **Command-line Interface:**
  ```powershell
  .\.venv\Scripts\python.exe -m experiments.MNIST_experiments.run_mnist_controlled --seeds 42 123 456 789 999 --experiments baseline_ann bnn_bp bnn_dfa dfa_least_squares dfa_binary_head dfa_direct_binary_head dfa_binary_head_ste bnn_dfa_qubo --output data/mnist_controlled_results.csv
  ```
- **Arguments:**
  - `--seeds`: List of integer seeds (default: `42 123 2024 7 99`; for paper reproduction specify: `42 123 456 789 999`).
  - `--experiments`: Subset or all of `{baseline_ann, bnn_bp, bnn_dfa, dfa_least_squares, dfa_binary_head, dfa_direct_binary_head, dfa_binary_head_ste, bnn_dfa_qubo}`.
  - `--output`: Output CSV path for raw run results (default: `data/mnist_controlled_results.csv`).
  - `--summary-output`: Output CSV path for aggregated statistics (mean, std, min, max; default: `<output-stem>_summary.csv` beside raw CSV).
  - `--overwrite`: Allow replacing existing output files.

#### 2. Shared-Representation Validation Runner: `run_mnist_shared_validation.py`
- **Purpose:** Compares multiple classifier heads on a **single, shared, frozen BNN-DFA backbone** per seed, with strict validation-based checkpointing and hyperparameter selection.
- **Protocol:**
  - Official MNIST training set (N = 60,000) is partitioned into a training partition (N_train = 51,000, 85%) and a stratified validation partition (N_val = 9,000, 15%).
  - One BNN-DFA backbone is trained for `--epochs` (default: 16) and saved/reused from `data/mnist_representations/`.
  - Heads compared on identical features H in R^(N x 128):
    1. **Ridge LS:** Closed-form solve on H_train.
    2. **Binarized Ridge LS:** Feature signs sign(W_ridge) with scale and bias refit on H_train.
    3. **QUBO Binary:** Validation-tuned over a 27-configuration grid (or `--fixed-qubo`), with post-annealing scale/bias fit on H_train.
    4. **STE Binary:** Trained for `--ste-epochs` (default: 50); validation accuracy selects the best training epoch checkpoint.
  - Official MNIST test set (N_test = 10,000) is evaluated **only once** for final reporting.
- **Command-line Interface:**
  ```powershell
  # One-seed pilot, with isolated cache and result output
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $env:BNN_DFA_RESULTS_CSV = "data/results_diagnostic/mnist_shared_validation_pilot_seed_42_$stamp.csv"
  .\.venv\Scripts\python.exe -m experiments.MNIST_experiments.run_mnist_shared_validation --seed 42 --epochs 16 --validation-fraction 0.15 --representation-dir data/mnist_representations/pilot_seed_42
  Remove-Item Env:BNN_DFA_RESULTS_CSV

  # Or with historical fixed 100 reads (no validation grid tuning)
  .\.venv\Scripts\python.exe -m experiments.MNIST_experiments.run_mnist_shared_validation --seed 42 --epochs 16 --fixed-qubo --num-reads 100 --representation-dir data/mnist_representations/pilot_seed_42
  ```
- **Prerequisites:** Run from the repository root with the project environment and dependencies installed (for Windows PowerShell, `.\.venv\Scripts\python.exe`). The first run downloads MNIST only if it is not already present under `data/MNIST/`; network access is therefore needed only for that initial dataset acquisition. No special accelerator is required by the CLI.
- **Representation cache:** `--representation-dir` is a directory containing a cache keyed by seed, backbone epochs, and validation fraction. Compatible artifacts are reused after checking dataset fingerprint, split indices, seed, backbone/training settings, tensor checksums, and label alignment. Git/package provenance and the artifact-generation device remain bound to the saved artifact ID but do not invalidate otherwise compatible features. A missing artifact is trained and saved with create-only semantics.
- **Cache recovery:** An incompatible or corrupted artifact is never overwritten or silently accepted. The error lists metadata fields that differ and prints a command using a fresh `--representation-dir`. For example, keep the original cache and retry with `--representation-dir data/mnist_representations/pilot_seed_42_rebuild`. You can also use the fresh directory printed by the error. Leave the old artifact in place for inspection.
- **Pilot versus study:** The command above is one seed for onboarding and pipeline validation. It is not the final multi-seed research protocol. For a study, run the prespecified seeds with a separate seed-specific representation directory for each seed, retain raw per-seed outputs, and aggregate only compatible runs.
- **Outputs Produced:**
  - Diagnostic comparison table: `data/results_diagnostic/shared_validation_heads_<protocol>_seed_<seed>_<timestamp>.csv`
  - QUBO tuning log: `data/results_diagnostic/qubo_validation_tuning_27config_seed_<seed>_<timestamp>.csv`
  - Automatic result append: Automatically appends the selected QUBO head result to `data/results/mnist_results.csv` via `results_utils.append_result`.

#### 3. QUBO Validation-Only Tuning: `tune_qubo_validation.py`
- **Purpose:** Performs a bounded grid search over QUBO solver hyperparameters using only training representations for optimization and validation representations for candidate selection.
- **Hyperparameter Grid (27 configurations):**
  - `num_reads`: `(25, 50, 100)`
  - `num_sweeps`: `(250, 500, 1000)`
  - `scale_multiplier`: `(0.5, 1.0, 2.0)`
  - Deterministic solver seed rule for entry i (0-indexed): `(backbone_seed + 1009 * (i + 1)) mod 2**32`.
- **Selection & Tie-Breaking Criteria:**
  Candidates are ranked by:
  1. Highest validation accuracy
  2. Fewest `num_reads`
  3. `scale_multiplier` closest to `1.0`
  4. Fewest `num_sweeps`
  5. Lowest `scale_multiplier`
- **Command-line Interface:**
  ```powershell
  .\.venv\Scripts\python.exe -m experiments.MNIST_experiments.tune_qubo_validation --seed 42 --epochs 16
  ```
- **Outputs Produced:**
  - `data/results_diagnostic/qubo_validation_tuning_27config_seed_<seed>_<timestamp>.csv`: Detailed evaluation of all 27 candidate configurations.
  - `data/results_diagnostic/qubo_validation_tuning_27config_seed_<seed>_<timestamp>.json`: Summary configuration metadata, winner parameters, validation score, and single final test accuracy.
  - *Note:* Standalone execution of `tune_qubo_validation.py` saves diagnostic files but does not append to `data/results/mnist_results.csv` (use `run_mnist_shared_validation.py` or `bnn_dfa_qubo.py --qubo-tuning` to log to the main results table).

#### 4. Shared Representation Management: `mnist_representation.py` & `utils/MNIST/representation.py`
- **Purpose:** Caches and verifies frozen BNN-DFA hidden feature activations (H_train in R^(51000x128), H_validation in R^(9000x128), H_test in R^(10000x128)).
- **Artifact Location:** `data/mnist_representations/mnist_bnn_dfa_seed_<seed>_epochs_<epochs>_val_<val_fraction>.pt`
- **Validation:** On load, verifies full metadata compatibility (dataset fingerprint SHA-256, stratified split indices, parameter counts, and tensor checksums). If incompatible or corrupt, raises an error; if valid, safely reuses the representation.

#### 5. Additional Shared Diagnostics:
- `compare_ls_binary_qubo_shared.py`: Compares Ridge LS, Binarized Ridge LS, and QUBO Binary on a single representation (`--seed`, `--epochs`, `--num-reads`, `--fixed-qubo`).
- `sweep_qubo_num_reads_shared.py`: Evaluates the impact of different QUBO read counts (`--reads 25 50 100 200`) across seeds on shared frozen features. Saves detailed and aggregate CSVs to `data/results_diagnostic/`.
- `results_utils.py`: Central helper managing schema upgrades and thread-safe writes to `data/results/mnist_results.csv` (redirectable via `$env:BNN_DFA_RESULTS_CSV`).

#### Computational-Cost Analysis and Timing Definitions

The shared-validation runner (`run_mnist_shared_validation.py`) measures the backbone, feature extraction, and each classifier head separately, providing transparent cost accounting across cached and fresh runs:

- **Backbone and Feature Extraction Costs:**
  - `backbone_training_time_sec`: Time in seconds spent training the BNN+DFA backbone (16 epochs). If the representation artifact is freshly trained (`representation_cache_status == "trained"`), this records the current run's measured training duration. If the artifact is reused from cache (`"reused"`), this field is `0.0` for current-run work, while the historical backbone training time remains preserved in the representation artifact metadata (`training_seconds`).
  - `feature_extraction_time_sec`: Time in seconds spent running forward passes through the frozen first layer ($784 	o 128$) to extract ReLU activations for train ($51{,}000$), validation ($9{,}000$), and test ($10{,}000$) partitions. When loading a cached artifact, extraction is bypassed and `0.0` is recorded for the current run.
  - `representation_cache_status`: Explicitly marks whether the feature representation was newly `"trained"` or `"reused"` from disk cache.

- **Per-Head Fitting Time (`head_fit_time_sec`):**
  Monotonically times (`time.perf_counter()`) only the fitting or optimization operations of each classifier head on CPU/GPU (with appropriate CUDA synchronization where applicable). **Excludes** test set scoring, validation set final evaluations, or diagnostic logging:
  - **Ridge LS:** SVD-based linear solve with float64 augmented system and unpenalized intercept on training features ($H_{	ext{train}}$).
  - **Binarized Ridge LS:** Feature weight sign binarization plus closed-form training-only scale and bias refitting.
  - **QUBO Binary:** The complete QUBO search and tuning loop across all 27 candidate configurations (simulated annealing on training Ising objectives, candidate bias recovery, and validation accuracy evaluation) plus final scale/bias fitting. For fixed-QUBO mode, times annealing of the single fixed read count and scale/bias refit.
  - **STE Binary:** Full straight-through estimator training loop (50 Adam epochs on $H_{	ext{train}}$) including validation-based best-epoch checkpoint selection.

- **Total Computational Cost Semantics:**
  - `current_run_total_time_sec`: Total wall-clock time of work actually executed during the current run:
    `current_run_total_time_sec = current_backbone_time + current_extraction_time + head_fit_time_sec`
    For cached representation runs, `current_backbone_time` and `current_extraction_time` are genuinely `0.0`, strictly isolating current-run head optimization without falsely attributing past backbone training to the current invocation.
  - `end_to_end_training_time_sec`: Component-based total pipeline cost estimate:
    `end_to_end_training_time_sec = artifact_training_time + artifact_extraction_time + head_fit_time_sec`
    Provides the full estimated lifecycle cost of training the feature extractor plus fitting the head.
    *Missing timing handling:* If historical backbone training or feature extraction durations are missing/unavailable in a loaded representation artifact, `end_to_end_training_time_sec` is recorded as blank/empty string (`""`) rather than silently substituting `0.0`, and the missing state is audited in `pipeline_timing_status`.
  - `pipeline_timing_status`: Audit field indicating whether all historical pipeline components were recorded (`"complete"`, `"missing_extraction_time"`, or `"missing_backbone_and_extraction_time"`).
  - Legacy CSV fields (`training_time_sec` and `parameter_count`):
    - `training_time_sec`: In the canonical results CSV, historical runs recorded backbone training loop time. For shared-validation rows, `training_time_sec` records current-run backbone training time (`0.0` when reused from cache, preserving compatibility with timing aggregation scripts).
    - `parameter_count`: Retains legacy total network backbone count (101,770) for compatibility with historical reporting tools, while `backbone_parameter_count`, `head_parameter_count`, and `total_parameter_count` provide the unbundled breakdown.
- **Parameter Counts:**
  - `backbone_parameter_count`: Always $101{,}770$ for the standard $784 	o 128 	o 10$ network ($784 	imes 128 + 128 = 100{,}480$ in layer 1; $128 	imes 10 + 10 = 1{,}290$ in layer 2).
  - `head_parameter_count`:
    - *Ridge LS:* $1{,}290$ ($128 	imes 10$ float weights $+ 10$ unpenalized intercepts).
    - *Binarized Ridge LS:* $1{,}300$ ($128 	imes 10 = 1{,}280$ binary weights $\{-1, +1\} + 10$ positive class scales $+ 10$ class biases).
    - *QUBO Binary:* $1{,}300$ ($128 	imes 10 = 1{,}280$ binary spin variables $\{-1, +1\} + 10$ continuous class scales $+ 10$ continuous biases).
    - *STE Binary:* $1{,}290$ deployed parameters ($128 	imes 10 = 1{,}280$ binary weights $+ 10$ learned scalar biases; optimizer-only latent momentum states are excluded from deployed parameter counts).
  - `total_parameter_count`: $	ext{backbone\_parameter\_count} + 	ext{head\_parameter\_count}$ ($103{,}060$ for Ridge LS and STE; $103{,}070$ for Binarized Ridge LS and QUBO Binary).

#### Protocol and integrity notes

- The individual MNIST scripts are independent end-to-end runs; `run_mnist_shared_validation.py` is a head-only comparison on one frozen, split-aligned representation per seed. Stroke `run_stroke_controlled.py` trains one shared representation per model seed and compares heads; `stroke_binary_head_multiseed.py` varies head seeds on a fixed representation. Do not combine these protocols in one mean or claim a shared seed proves identical features.
- MNIST `tune_qubo_validation.py` is the selection workflow: its fixed 27 candidates are fit on training representations, ranked by validation accuracy (then reads, multiplier distance, sweeps, multiplier), and the test set is scored once after selection. `sweep_qubo_num_reads_shared.py` reports test accuracy for each read count without validation selection; those metrics are post-hoc descriptive only and must not choose reads or support unbiased selected-model claims.
- MNIST QUBO `qubo_time_sec` on new tuned rows is the complete tuner duration through candidate fitting/refitting, validation scoring, and winner selection. `qubo_selected_candidate_time_sec` is the winner's candidate duration. Test evaluation and shared representation setup are outside both fields. Candidate CSV rows retain status and runtime; all-failed grids are saved before the runner stops.
- New Stroke representations use schema v3, content SHA-256 over canonical tensor bytes plus shape/dtype and protocol metadata, and persisted split row IDs and labels. v2 artifacts load only with an explicit `legacy_unverified` warning. Exact cross-platform numerical reproducibility is not claimed; run metadata records Git state, Python/packages, platform/device, and stable run IDs where available.
- Stroke seed variation uses one fixed split, so it measures seed variability and not split-to-split generalization. Stroke thresholds and candidate selection are validation-derived; test metrics are final evaluation only.
- The historical MNIST seed-123 QUBO row remains unchanged. Its approximately `0.0000069` second `qubo_time_sec` is not supported by candidate-level timings and is excluded from efficiency comparisons pending independent substantiation.
- Aggregate writers preserve protocol and selected-configuration groups and refuse to replace existing tables. Use a new output path or explicit overwrite option where the runner provides one.

---

### 2.4 Explaining the Paper's MNIST Tables and Figures

The paper reports MNIST results across five random seeds: `42, 123, 456, 789, 999`.

#### Table III: Mean MNIST Test Accuracy and Timing Across Five Seeds
- **Content:** Mean +/- standard deviation of test accuracy, mean training time (backbone training loop in seconds), and mean QUBO optimization time for the eight methods.
- **Underlying Data:** Recorded in `Old_results/mnist_results.csv` (and reproduced in `data/results/mnist_results.csv` if historical experiments are run).
  - Standard ANN: 97.74 +/- 0.10% (mean train time: 134.76 s)
  - BNN + Backpropagation: 95.35 +/- 0.62% (mean train time: 141.17 s)
  - BNN + DFA: 92.69 +/- 1.17% (mean train time: 111.12 s)
  - BNN + DFA + Real-Valued LS Head: 48.10 +/- 39.62% (mean train time: 108.48 s)
  - BNN + DFA + Binarized LS Head: 38.30 +/- 28.36% (mean train time: 111.41 s)
  - BNN + DFA + Direct Binary Head: 70.46 +/- 7.19% (mean train time: 114.68 s)
  - BNN + DFA + STE Binary Head: 13.68 +/- 3.54% (mean train time: 112.16 s)
  - BNN + DFA + QUBO: 89.56 +/- 4.19% (mean train time: 106.39 s, mean QUBO time: 3.55 s)
- **Automated Generation:** Supported via `experiments/MNIST_experiments/collect_results.py`.
  ```powershell
  .\.venv\Scripts\python.exe -m experiments.MNIST_experiments.collect_results
  ```
  *Outputs:* Generates `data/results/main_classification_results.csv` (mean +/- std) and `data/results/computational_cost_results.csv` from `data/results/mnist_results.csv`.

#### Table IV: MNIST Test Accuracy (%) by Seed
- **Content:** Per-seed test accuracy across the five seeds (`42, 123, 456, 789, 999`) for all eight methods:

| Seed | ANN | BNN+BP | BNN+DFA | Real LS | Bin. LS | Direct | STE | QUBO |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 97.88 | 95.89 | 93.92 | 19.12 | 14.91 | 69.28 | 18.98 | 93.40 |
| **123** | 97.65 | 95.60 | 91.35 | 29.81 | 34.36 | 74.79 | 13.31 | 93.04 |
| **456** | 97.77 | 95.56 | 93.65 | 10.10 | 9.80 | 76.68 | 9.22 | 83.27 |
| **789** | 97.74 | 95.40 | 92.92 | 87.11 | 78.14 | 72.99 | 14.39 | 90.33 |
| **999** | 97.65 | 94.28 | 91.61 | 94.34 | 54.30 | 58.56 | 12.51 | 87.74 |

- **Underlying Data:** `Old_results/mnist_results.csv` (or `data/results/mnist_results.csv`).
- **Generation:** Can be inspected by filtering/pivoting `mnist_results.csv`, or generated automatically from a multi-seed controlled run via `plot_mnist_results.py`:
  ```powershell
  .\.venv\Scripts\python.exe -m experiments.MNIST_experiments.plot_mnist_results
  ```
  *(Outputs `data/mnist_results/test_accuracy_per_seed.png` and `data/mnist_results/mnist_results_summary.csv` from `data/mnist_controlled_results.csv`).*

#### Figure 5: MNIST Test Accuracy Across Five Seeds (Mean +/- SD)
- **Content:** Bar chart showing test accuracy mean +/- standard deviation error bars across the five seeds for all eight methods.
- **Automated Generation:** Supported via `experiments/MNIST_experiments/plot_results.py`.
  ```powershell
  .\.venv\Scripts\python.exe -m experiments.MNIST_experiments.plot_results
  ```
  *Outputs:* Generates `data/results/figures/fig1_main_classification_accuracy.png` (matching Figure 5), `fig2_classifier_head_comparison.png`, `fig3_qubo_comparison.png`, and `fig5_accuracy_vs_training_time.png` directly from `data/results/mnist_results.csv`.

---

### 2.5 Windows PowerShell Quick Start

Follow these verified steps in Windows PowerShell from the repository root:

#### 1. Environment & Installation Check
```powershell
# Verify Python executable and version
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable); print(sys.version)"

# Verify core dependencies
.\.venv\Scripts\python.exe -c "import torch, torchvision, numpy, pandas, neal; print('Imports OK, PyTorch:', torch.__version__)"

# Run existing unit tests for MNIST experiments, schemas, and representation invariants
.\.venv\Scripts\python.exe -m unittest discover tests
```

#### 2. One-Seed Smoke Test (Isolated Diagnostic Directory)
To avoid overwriting historical results, redirect output logging using `$env:BNN_DFA_RESULTS_CSV`:
```powershell
# Set up a timestamped diagnostic output file
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$env:BNN_DFA_RESULTS_CSV = "data/results_diagnostic/smoke_mnist_$stamp.csv"

# Run a quick smoke test for an end-to-end baseline and a head
.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.baseline_ann --seed 42
.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.dfa_least_squares --seed 42 --ls-solver legacy
.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.bnn_dfa_qubo --seed 42 --qubo-scale-solver legacy --num-reads 100

# Remove the redirection variable after the run
Remove-Item Env:BNN_DFA_RESULTS_CSV
```

#### 3. Multi-Seed Reproduction Command (Paper's Five Seeds)
To run the full eight-method suite across the paper's five seeds (`42 123 456 789 999`) using `run_mnist_controlled.py`:
```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.run_mnist_controlled `
    --seeds 42 123 456 789 999 `
    --experiments baseline_ann bnn_bp bnn_dfa dfa_least_squares dfa_binary_head dfa_direct_binary_head dfa_binary_head_ste bnn_dfa_qubo `
    --output "data/results_diagnostic/mnist_paper_runs_$stamp.csv"
```
*Note: This command runs 40 training executions (8 methods x 5 seeds) and saves raw and summary CSVs to `data/results_diagnostic/`.*

#### 4. Shared-Representation Validation Comparison
For the one-seed isolated pilot command, see Section 2.3 above.

#### 5. QUBO Validation-Tuning Grid Search (Standalone)
To run the 27-configuration tuning grid (3 x 3 x 3) on a shared representation:
```powershell
.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.tune_qubo_validation --seed 42 --epochs 16
```

#### 6. Generate Paper Tables and Figures
```powershell
# Generate summary CSVs from data/results/mnist_results.csv
.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.collect_results

# Generate Figure 5 and head comparison plots from data/results/mnist_results.csv
.\.venv\Scripts\python.exe -m experiments.MNIST_experiments.plot_results
```

#### 7. Inspect Generated Results
```powershell
# View summary table produced by collect_results
Get-Content -Path data\results\main_classification_results.csv | Format-Table

# View summary produced by run_mnist_controlled
Get-Content -Path data\mnist_controlled_results_summary.csv | Format-Table
```

### 2.6 Output Locations and Protocol Distinction

- **Historical Paper Results (`Old_results/mnist_results.csv`):** Contains the 40 original runs across seeds `42, 123, 456, 789, 999` using full 60,000-image training sets, unregularized legacy float32 LS, and independently retrained BNN-DFA backbones.
- **Main Local Run Log (`data/results/mnist_results.csv`):** Default append target for individual experiment scripts and selected QUBO shared-validation runs.
- **Diagnostic Result Directory (`data/results_diagnostic/`):** Contains timestamped CSV and JSON artifacts for QUBO 27-configuration grid tuning, shared-validation head comparisons, and smoke runs.
- **Representation Cache (`data/mnist_representations/`):** Caches frozen BNN-DFA feature tensors (`mnist_bnn_dfa_seed_<seed>_epochs_<epochs>_val_<val_fraction>.pt`) with metadata and SHA-256 integrity verification.
- **Controlled Runner Outputs (`data/mnist_controlled_results.csv` and `data/mnist_controlled_results_summary.csv`):** Output from `run_mnist_controlled.py`.

> [!CAUTION]
> **Do not confuse the protocols:**
> - **Full-data independent backbone protocol:** 60,000 train, 10,000 test. Each method retrains its own feature extractor.
> - **Shared-validation protocol:** 51,000 train, 9,000 validation, 10,000 test. All heads share the exact same frozen features from one BNN-DFA backbone. Validation data is strictly held out from training and used to tune QUBO parameters and pick the best STE epoch.

## 3. Stroke experiments

### Download and prepare the data

The downloader retrieves the public Fedesoriano Stroke Prediction Dataset through KaggleHub. A CSV already present locally is protected from accidental replacement.

```powershell
python -m experiments.stroke_experiments.download_stroke
python -m experiments.stroke_experiments.process_stroke --seed 42 --balancing random_oversample
```

The preprocessing flow creates stratified 70/15/15 train/validation/test splits, fits learned preprocessing on the training split only, and applies optional balancing to training rows only. Validation and test remain imbalanced. The saved metadata records the raw-data hash, split seed, feature names/order, preprocessing configuration, class counts, and other provenance information.

The current Stroke preprocessing version is `finalboss-stroke-v3-scaled-inputs`. It keeps age, glucose, BMI, and the age–glucose interaction as train-fitted standardized Yeo–Johnson features; it does not feed their large raw-valued counterparts directly to the BNN. This keeps input magnitudes more comparable and prevents the unscaled interaction from dominating the first binary layer. The derived clinical flags and categorical encodings are preserved. Any representation or preprocessing artifacts from an older version must be regenerated before running the controlled comparison.

The standard paths are controlled by these environment variables:

- `STROKE_DATA_DIR`: local directory for preprocessing artifacts and representations; defaults to `data/stroke`.
- `STROKE_CSV`: path to the raw Stroke CSV; defaults to `data/stroke/healthcare-dataset-stroke-data.csv`.
- `STROKE_SPLIT_SEED`: default split seed; defaults to `42`.
- `STROKE_BALANCING_METHOD`: default training balancing method; defaults to `random_oversample`.

### Run the complete controlled comparison

```powershell
python -m experiments.stroke_experiments.run_stroke_controlled --split-seed 42 --seeds 42 123 2024 7 99 --balancing random_oversample --ls-solver both
```

Each model seed trains a DFA representation; all classifier heads for that seed use the same saved representation. Metrics include balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion-matrix counts. The output records the representation seed, split seed, data/preprocessing provenance, target encoding, objective, and bias treatment. With `--ls-solver both`, the legacy and Ridge-derived binarized LS heads are evaluated on the same representation per seed. The CSV includes `status` and `failure_reason` columns so a detected degenerate QUBO head is recorded explicitly and later seeds can still run.

**Rerun protection:** the controlled runner refuses to replace existing representation, model, or result files unless `--overwrite` is supplied. Only use `--overwrite` when you intend to replace those artifacts. For isolated runs, pass a new nested `--output-dir`, for example `data/stroke_runs/run_20261014/stroke`; the runner stores its CSV beside that output directory.

### Stroke LS variants

The QUBO head calibrates its initial score scale using the shared float64 SVD Ridge solver on the same signed `{-1,+1}` target encoding used by its QUBO objective. It fits the hidden-feature standardizer on training representations only and searches a configured alpha-multiplier grid using simulated annealing (recorded reads and sweeps per scale). In addition to low-energy annealer samples and their coordinate refinements, each scale evaluates class-mean, Ridge-derived, and reproducible random binary starts refined using the training-derived Ising energy. Candidate selection uses an equal-weight validation composite of balanced accuracy, F1, PR-AUC, and ROC-AUC; composite ties favor balanced accuracy. The test split is not used for calibration, candidate construction, ranking, or threshold selection. The result CSV records alpha settings, candidate source/count, energy, reads, sweeps, random-start budget, and validation metrics. Degenerate representations raise a clear error instead of silently returning an all-positive classifier.

The stable binary-weight heads (Direct Binary, Ridge-derived Binarized LS, STE, and QUBO) use the same train-fitted hidden-feature standardization convention. Legacy Binarized LS intentionally retains the raw representation for reproducibility. The Stroke STE head divides its BCE-with-logits scores by `sqrt(hidden_size)` during optimization to limit activation-scale-driven gradients. It tests Ridge, class-mean, and random training-derived directions, then uses confidence-weighted latent initial magnitudes: weaker direction components start closer to zero so straight-through gradients can flip their binary signs, while more confident components start farther from the sign boundary. It evaluates each initial binary vector as an epoch-zero baseline, retains the best validation checkpoint across all subsequent sign patterns, and uses patience-based stopping. The best candidate is selected by the equal-weight validation composite of balanced accuracy, F1, PR-AUC, and ROC-AUC (ties favor balanced accuracy). The result CSV records the selected epoch and how many binary signs changed from the start, allowing us to verify that STE actually optimized rather than merely falling back to the initial classifier.

The existing Stroke Real-LS path is an unregularized NumPy least-squares fit with `{0,1}` targets. The Stroke Binarized Ridge-LS path uses the shared float64 augmented SVD solver, signs the feature weights, and refits the scale and intercept using training representations only.

**Result-version note:** earlier five-seed comparisons were generated with preprocessing v2 and the old 23-feature input schema. Preprocessing v3 changes the input schema to 20 features and rescales continuous inputs, so those older metrics and head rankings must not be presented as the v3 results. Regenerate the complete controlled run and its summary under v3 before updating the paper or drawing conclusions about the current models.

For a saved representation:

```powershell
# Existing legacy binarized LS implementation
python -m experiments.stroke_experiments.stroke_dfa_binarized_ls --representation data/stroke/stroke_dfa_representation_seed_42.pt --ls-solver legacy

# Candidate stable Ridge-LS weights and refitted scale/bias
python -m experiments.stroke_experiments.stroke_dfa_binarized_ls --representation data/stroke/stroke_dfa_representation_seed_42.pt --ls-solver ridge --lambda-relative 1e-4
```

The controlled runner supports the same choice:

```powershell
# Default: validated Ridge-derived binary LS head
python -m experiments.stroke_experiments.run_stroke_controlled --split-seed 42 --seeds 42 123 2024 7 99 --ls-solver ridge

# Reproduce the previous binarized-LS implementation
python -m experiments.stroke_experiments.run_stroke_controlled --split-seed 42 --seeds 42 123 2024 7 99 --ls-solver legacy
```

### Paired LS-head validation

This diagnostic compares four LS variants on the same saved representation for each model seed: legacy Real LS, Ridge LS, legacy Binarized LS, and Binarized Ridge LS with scale/bias refitting. The validation split selects the decision threshold; the held-out test split is used only for final metrics. It produces timestamped detailed and aggregate CSV files under `data/results_diagnostic/` and does not overwrite the main Stroke result file.

```powershell
# Run synthetic tests for the solver and LS-head implementations
python -m unittest experiments.stroke_experiments.test_stroke_ls_heads

# Reuse existing compatible representations
python -m experiments.stroke_experiments.validate_stroke_ls_heads --representation-dir data/stroke --seeds 42 123 2024 7 99 --split-seed 42 --balancing random_oversample

# Or train fresh representations and save them to a new timestamped artifact directory
python -m experiments.stroke_experiments.validate_stroke_ls_heads --seeds 42 123 2024 7 99 --split-seed 42 --balancing random_oversample
```

Threshold selection evaluates the unique validation-score boundaries, so large gaps or extreme scores do not cause a fixed-grid search to skip useful thresholds. If this diagnostic script changes, rerun it and keep the resulting CSVs distinct; the earlier Stroke validation CSV was generated using a coarser threshold grid and should not be treated as the final comparison.

### Generate a new Stroke results bundle and plots

The following PowerShell example saves the representations, controlled results CSV, and plots under a unique timestamped directory. It includes both binarized LS variants while training only one BNN+DFA representation per seed:

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = "data/stroke_runs/run_$stamp"
$artifactDir = Join-Path $runRoot "stroke"
$resultsCsv = Join-Path $runRoot "stroke_controlled_results.csv"
$plotsDir = Join-Path $runRoot "plots"

python -m experiments.stroke_experiments.run_stroke_controlled --output-dir $artifactDir --split-seed 42 --seeds 42 123 2024 7 99 --balancing random_oversample --ls-solver both

python -m experiments.stroke_experiments.plot_stroke_results --input $resultsCsv --output-dir $plotsDir
```

The plotting script also supports the default paths:

```powershell
python -m experiments.stroke_experiments.plot_stroke_results
```

It writes PNG charts for balanced accuracy, PR-AUC, ROC-AUC, and F1 plus a summary CSV. It refuses to overwrite an existing summary or plot unless `--overwrite` is supplied; choose a new `--output-dir` for a separate run.

### Lightweight remediation tests

These tests use synthetic inputs and do not download data or train full models:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_model_invariants tests.test_qubo_validation_tuning
.\.venv\Scripts\python.exe -m unittest tests.test_binarized_ridge_audit tests.test_mnist_representation tests.test_mnist_results_schema tests.test_model_invariants tests.test_remediation_integrity
.\.venv\Scripts\python.exe -m unittest experiments.MNIST_experiments.test_ste_head tests.test_direct_binary_head
.\.venv\Scripts\python.exe -m unittest experiments.stroke_experiments.test_stroke_qubo
.\.venv\Scripts\python.exe -m unittest experiments.stroke_experiments.test_stroke_preprocessing
.\.venv\Scripts\python.exe -m unittest experiments.stroke_experiments.test_stroke_ls_heads
```

## 4. Cross-platform reproducibility

The repository includes a fixed MNIST data-order protocol intended to control training example order when comparing CPU, CUDA, and Apple MPS runs. It helps isolate sources of variation; it does **not** guarantee bit-for-bit equality across different devices or library versions.

### Verify, then run

```powershell
python -m reproducibility.verify_protocol
python -m reproducibility.runner dfa_least_squares
python -m reproducibility.runner dfa_binary_head
python -m reproducibility.runner bnn_dfa_qubo
```

The canonical runner uses the checked-in `protocol/manifest.json`, fixed test indices, and predetermined per-epoch training permutations. Use the same checked-in protocol on every machine. The protocol generator refuses to replace existing protocol files by default. Do not regenerate the protocol on a second machine. Only regenerate intentionally on the designated machine with `python -m reproducibility.generate_protocol --overwrite`, then review and share/commit the manifest and all protocol files as one versioned change.

The reproducibility protocol currently targets the MNIST track; Stroke reproducibility instead relies on explicit model seeds, a fixed split seed, validated preprocessing artifacts, representation metadata, and saved dataset/configuration hashes. To strengthen a Stroke comparison, keep the dataset hash, split seed, balancing method, model seeds, preprocessing metadata, package versions, and result CSV together.

## 5. Results, plots, and repository layout

- `results/*.csv`: version-controlled historical/reference result tables.
- `data/results/mnist_results.csv`: default local log for individual MNIST experiments.
- `data/mnist_controlled_results.csv` and its summary: output from the controlled MNIST runner.
- `data/stroke_controlled_results.csv`: default output from the controlled Stroke comparison.
- `data/stroke_results/*.png`: Stroke metric plots.
- `data/results_diagnostic/`: timestamped diagnostic CSVs for solver/head validation.
- `data/stroke/`: raw Stroke CSV, preprocessing artifacts, metadata, and representations.

The `data/` directory is Git-ignored to avoid committing raw datasets, large representations, and generated results by accident. Back up or share the specific result CSVs and plots needed for a review rather than committing all local artifacts.

## 6. Troubleshooting

- **Missing raw data:** run the relevant dataset downloader or allow the MNIST loader to download MNIST.
- **Stale or mismatched Stroke metadata:** rerun preprocessing with the intended split seed and balancing configuration. Use `--overwrite` only after deciding to replace the saved artifacts.
- **FileExistsError from a controlled Stroke run:** the run is protecting existing results. Choose a new `--output-dir` or explicitly use `--overwrite` if replacement is intended.
- **Different accuracies across hardware:** verify the seed, protocol/split, dataset hash, package versions, and method settings first. Hardware-specific floating-point operations can still produce numerical differences.

## 7. Current LS validation status on the feature branch

The stable MNIST Ridge-LS path has been evaluated across multiple seeds. Historical Stroke v2 results (the old 23-feature input schema) showed Ridge-derived Binarized LS improving mean test balanced accuracy (71.59% vs. 63.35%) and mean ROC-AUC (78.26% vs. 55.65%) over legacy Binarized LS. Those numbers are **historical v2 results only**: v3 now standardizes the hidden inputs for the stable binary heads and adds STE initialization / QUBO scale candidate selection, so the v2 comparison is not evidence for the current ranking. Rerun the full controlled comparison under the exact v3 code before making current performance claims. The controlled Stroke runner defaults to `--ls-solver ridge`; use `--ls-solver legacy` to reproduce the legacy binarized-LS implementation or `--ls-solver both` to record both heads on each representation. The continuous Stroke Real-LS default remains unchanged.
