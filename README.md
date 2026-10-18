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

### Run individual models

Run each command from the repository root:

```powershell
# Standard ANN
python -m experiments.MNIST_experiments.baseline_ann --seed 42

# BNN trained with backpropagation
python -m experiments.MNIST_experiments.bnn_bp --seed 42

# BNN trained with Direct Feedback Alignment
python -m experiments.MNIST_experiments.bnn_dfa --seed 42

# DFA with a real-valued least-squares head
python -m experiments.MNIST_experiments.dfa_least_squares --seed 42

# DFA with a binarized LS-derived head
python -m experiments.MNIST_experiments.dfa_binary_head --seed 42

# Direct binary head
python -m experiments.MNIST_experiments.dfa_direct_binary_head --seed 42

# Straight-through estimator binary head
python -m experiments.MNIST_experiments.dfa_binary_head_ste --seed 42

# QUBO / Ising binary head
python -m experiments.MNIST_experiments.bnn_dfa_qubo --seed 42
```

The main classifier-head scripts currently use the numerically stable Ridge-LS implementation by default on this feature branch. Use the legacy switches when reproducing the original implementation:

```powershell
# Original float32 torch.linalg.lstsq implementation for real LS
python -m experiments.MNIST_experiments.dfa_least_squares --seed 42 --ls-solver legacy

# Original LS initializer / bias treatment for the binarized LS head
python -m experiments.MNIST_experiments.dfa_binary_head --seed 42 --ls-solver legacy

# Original QUBO scale initialization
python -m experiments.MNIST_experiments.bnn_dfa_qubo --seed 42 --qubo-scale-solver legacy
```

The Ridge-LS solver uses CPU float64 and an augmented SVD-based least-squares system, leaves the intercept unregularized, and avoids normal equations. The binarized Ridge-LS variant separately refits its class-specific scale and bias after weight sign binarization. Those are algorithmically distinct variants; compare the result labels instead of combining them under one generic LS name.

### Controlled multi-seed MNIST runs

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
python -m experiments.MNIST_experiments.run_mnist_controlled --seeds 42 123 456 789 999 --experiments dfa_least_squares dfa_binary_head bnn_dfa_qubo --output "data/results_diagnostic/mnist_controlled_$stamp.csv"
```

This controlled runner produces a per-run CSV and a matching summary CSV beside it. It refuses to replace existing outputs. Use a new `--output` path for each run, or pass `--overwrite` only when replacement is intentional. You can override the summary path with `--summary-output`.

### Keep diagnostic outputs separate

Most individual MNIST scripts append to `data/results/mnist_results.csv`. To send new diagnostic runs to another file, set `BNN_DFA_RESULTS_CSV` before launching the command.

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$env:BNN_DFA_RESULTS_CSV = "data/results_diagnostic/mnist_$stamp.csv"

python -m experiments.MNIST_experiments.dfa_least_squares --seed 42
python -m experiments.MNIST_experiments.dfa_binary_head --seed 42
python -m experiments.MNIST_experiments.bnn_dfa_qubo --seed 42

Remove-Item Env:BNN_DFA_RESULTS_CSV
```

This environment variable redirects those scripts' result logging. It does not redirect every custom diagnostic or the Stroke runner, which save their outputs separately.

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
python -m experiments.stroke_experiments.run_stroke_controlled --split-seed 42 --seeds 42 123 2024 7 99 --balancing random_oversample
```

Each model seed trains a DFA representation; all classifier heads for that seed use the same saved representation. Metrics include balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion-matrix counts. The output records the representation seed, split seed, data/preprocessing provenance, target encoding, objective, and bias treatment.

**Rerun protection:** the controlled runner refuses to replace existing representation, model, or result files unless `--overwrite` is supplied. Only use `--overwrite` when you intend to replace those artifacts. For isolated runs, pass a new nested `--output-dir`, for example `data/stroke_runs/run_20261014/stroke`; the runner stores its CSV beside that output directory.

### Stroke LS variants

The QUBO head calibrates its score scale using the shared float64 SVD Ridge solver on the same signed `{-1,+1}` target encoding used by its QUBO objective. The scale must be finite and non-negligible; degenerate representations now raise a clear error instead of silently returning an all-positive classifier. Stroke DFA checkpoints are selected using validation balanced accuracy over decision thresholds, rather than a hard-coded probability cutoff of 0.5.

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

The stable MNIST Ridge-LS path has been evaluated across multiple seeds. The corrected five-seed Stroke comparison selects thresholds using exact validation-score boundaries. Ridge-derived Binarized LS improved mean test balanced accuracy (71.59% vs. 63.35%) and mean ROC-AUC (78.26% vs. 55.65%) over legacy Binarized LS, with markedly lower seed-to-seed variation. The controlled Stroke runner therefore defaults to `--ls-solver ridge`; use `--ls-solver legacy` to reproduce the previous method or `--ls-solver both` to record both binary heads on each representation. The continuous Stroke Real-LS default remains unchanged because regularized Ridge LS did not improve its mean balanced accuracy.
