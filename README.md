BNN-DFA-QUBO

A research project investigating the combination of Binary Neural
Networks (BNNs), Direct Feedback Alignment (DFA), and a
QUBO/Ising-based binary classifier head on the MNIST dataset.

The repository contains a sequence of experiments that progressively
compare standard neural networks, BNNs trained with backpropagation,
BNNs trained using DFA, different binary classifier heads, and finally a
QUBO/Ising classifier optimized using simulated annealing.

Procedure to run the project:

1. Clone

git clone https://github.com/BNN-DFA-QUBO/BNN-DFA-QUBO.git
cd BNN-DFA-QUBO

2. Create and activate a virtual environment

Windows PowerShell

python -m venv venv
.\venv\Scripts\Activate.ps1

Windows Command Prompt

python -m venv venv
venv\Scripts\activate.bat

Linux / macOS

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

4. Verify the setup

python -c "import utils; import models; print('Project imports: OK')"
python -m experiments.test_data

MNIST downloads automatically into data/MNIST/raw/ when needed.

5. Run experiments

Run all commands from the repository root.

Standard ANN

python -m experiments.baseline_ann

BNN + Backpropagation

python -m experiments.bnn_bp

BNN + DFA

python -m experiments.bnn_dfa

DFA + Least-Squares Head

python -m experiments.dfa_least_squares

DFA + Binary Least-Squares Head

python -m experiments.dfa_binary_head

DFA + Tanh Binary Head

python -m experiments.dfa_direct_binary_head

DFA + STE Binary Head

python -m experiments.dfa_binary_head_ste

DFA + QUBO Binary Head

python -m experiments.bnn_dfa_qubo

6. Recommended order

python -m experiments.MNIST_experiments.test_data
python -m experiments.MNIST_experiments.baseline_ann
python -m experiments.MNIST_experiments.bnn_bp
python -m experiments.MNIST_experiments.bnn_dfa
python -m experiments.MNIST_experiments.dfa_least_squares
python -m experiments.MNIST_experiments.dfa_binary_head
python -m experiments.MNIST_experiments.dfa_direct_binary_head
python -m experiments.MNIST_experiments.dfa_binary_head_ste
python -m experiments.MNIST_experiments.bnn_dfa_qubo


To run an experiment with a specific seed:

python -m experiments.MNIST_experiments.baseline_ann --seed 123
python -m experiments.MNIST_experiments.bnn_bp --seed 123
python -m experiments.MNIST_experiments.bnn_dfa --seed 123
python -m experiments.MNIST_experiments.dfa_least_squares --seed 123
python -m experiments.MNIST_experiments.dfa_binary_head --seed 123
python -m experiments.MNIST_experiments.dfa_direct_binary_head --seed 123
python -m experiments.MNIST_experiments.dfa_binary_head_ste --seed 123
python -m experiments.MNIST_experiments.bnn_dfa_qubo --seed 123


7. Cross-platform reproducibility protocol

The repository includes a separate reproducibility layer for comparing
experiments across platforms such as NVIDIA CUDA, Apple MPS, and CPU.

7.1 Verify the protocol

Before running the experiments, verify that the protocol is complete and
valid:

python -m reproducibility.verify_protocol

The verification checks the manifest, test indices, and every training
epoch permutation.

7.2 Run an experiment with the reproducibility protocol

python -m reproducibility.runner baseline_ann
python -m reproducibility.runner bnn_bp
python -m reproducibility.runner bnn_dfa
python -m reproducibility.runner dfa_least_squares
python -m reproducibility.runner dfa_binary_head
python -m reproducibility.runner dfa_direct_binary_head
python -m reproducibility.runner dfa_binary_head_ste
python -m reproducibility.runner bnn_dfa_qubo

The purpose of this procedure is to isolate platform-dependent numerical
differences by controlling the experimental inputs and training order.
It does not guarantee bit-for-bit identical training results across
different hardware backends. Small numerical differences can still occur
because CUDA, MPS, and CPU use different kernels and floating-point
execution paths.


Project structure

BNN-DFA-QUBO/
├── experiments/
├── models/
├── utils/
├── reproducibility/
│   ├── deterministic.py
│   ├── fixed_data.py
│   ├── generate_protocol.py
│   ├── protocol_loader.py
│   ├── runner.py
│   └── verify_protocol.py
├── protocol/
│   ├── manifest.json
│   ├── test_indices.npy
│   └── train/
├── data/
├── pyproject.toml
├── requirements.txt
└── README.md

## Stroke prediction experiments

The stroke pipeline downloads the public Fedesoriano Kaggle dataset (`fedesoriano/stroke-prediction-dataset`) through KaggleHub. Public dataset downloads do not ordinarily need Kaggle credentials. A raw CSV already present at `data/stroke/healthcare-dataset-stroke-data.csv` is preserved by default; replacing it requires the explicit `--overwrite` flag.

Run these commands from the repository root after installing `requirements.txt`:

```powershell
# Download; refuses to replace an existing local CSV.
python -m experiments.stroke_experiments.download_stroke
# To replace an existing local CSV intentionally, use:
# python -m experiments.stroke_experiments.download_stroke --overwrite

# Create deterministic stratified 70/15/15 splits, fit preprocessing, and balance train only.
python -m experiments.stroke_experiments.process_stroke --seed 42
# Optional: write preprocessing artifacts to a specific directory.
# python -m experiments.stroke_experiments.process_stroke --seed 42 --output-dir data/stroke_run_42

# BNN trained with backpropagation (ordinary unweighted mean BCEWithLogitsLoss).
python -m experiments.stroke_experiments.stroke_bnn_bp

# Fixed split, multiple BNN/DFA training seeds, then all five heads per representation.
python -m experiments.stroke_experiments.run_stroke_controlled --split-seed 42 --seeds 42 123 2024 7 99
# To intentionally replace generated representations/results on a rerun, add --overwrite.
# python -m experiments.stroke_experiments.run_stroke_controlled --split-seed 42 --seeds 42 123 2024 7 99 --overwrite

# Individual DFA representation and head workflows (after process_stroke).
python -m experiments.stroke_experiments.stroke_bnn_dfa
python -m experiments.stroke_experiments.stroke_real_ls
python -m experiments.stroke_experiments.stroke_dfa_binarized_ls
python -m experiments.stroke_experiments.stroke_dfa_direct_binary
python -m experiments.stroke_experiments.stroke_dfa_ste_binary
python -m experiments.stroke_experiments.stroke_dfa_qubo
python -m experiments.stroke_experiments.stroke_dfa_least_squares

# Optional multi-seed binary-head comparison.
python -m experiments.stroke_experiments.stroke_binary_head_multiseed

# Run focused scientific-invariant tests (synthetic fixtures; no download required).
python -m unittest experiments.stroke_experiments.test_stroke_preprocessing
```

The data directory is ignored by Git. Set `STROKE_DATA_DIR` to choose another local data/artifact directory, `STROKE_CSV` to point at an existing raw CSV, or `STROKE_SPLIT_SEED` to set the default split seed. The split seed is separate from each BNN/DFA model seed: use `--split-seed` to vary partitions, and `--seeds` to compare model-training seeds on one fixed partition. The controlled runner prepares the requested split before training. It writes `metadata.json` (raw SHA-256, split policy and seed, row IDs per split, feature order, schema/config hashes, runtime library versions, and dynamic class counts) and `preprocessor.joblib` under `STROKE_DATA_DIR`. Matching artifacts are reused only after validation. Stale or incompatible artifacts fail clearly; pass `process_stroke --overwrite` or the controlled runner's `--overwrite` to regenerate them. Existing result files are also protected from accidental replacement. `process_stroke --output-dir` selects a specific artifact directory and reports that path.

Preprocessing follows the FinalBoss definitions and feature order: binary gender/marital/residence fields; age, hypertension, heart disease, glucose, BMI; cardiovascular comorbidity; raw age/glucose interaction; glucose risk tier; metabolic syndrome flag; grouped-work and smoking one-hot columns; and Yeo–Johnson versions of glucose, BMI, and the interaction. KNN imputation is fitted on training rows using age, gender, glucose, hypertension, heart disease, and BMI. These inputs remain on their raw scales to match the supplied specification; Euclidean distance therefore gives larger-range glucose, age, and BMI values more influence than 0/1 indicators. No rescaling is introduced without a supporting project specification. Input width comes from the fitted feature names (23 columns for the supplied dataset), not a model constant. The single `gender == Other` row is excluded before splitting while its ID remains auditable in the source file. All learned imputers, encoders, and power transforms are fitted on train only and reused unchanged for validation/test.

Balancing is explicit: `random_oversample` duplicates minority training rows exactly and does not synthesize records. The supplied FinalBoss script uses ordinary SMOTE followed by rounding encoded category flags, which can make invalid multi-hot combinations; the project therefore retains exact-row random oversampling as its documented alternative. Original and post-balancing counts and method are stored in metadata. Validation and test are never resampled and retain their imbalanced class distribution. Set `STROKE_BALANCING_METHOD=none` or pass `--balancing none` to compare without balancing. BNN backprop and DFA use ordinary mean `BCEWithLogitsLoss` with no `pos_weight`; the DFA output error is scaled by batch size to match that mean reduction.

Saved representations include training seed, split seed, preprocessing/schema version, raw-data hash, config hash, feature names, and aligned labels/row IDs. Old or incompatible representations are rejected by head loaders. The `run_stroke_controlled` output is an end-to-end comparison: each model seed trains one DFA representation, and all five heads for that seed use it and its saved labels. Its methods are distinct variants: Real LS fits `{0,1}` targets with an intercept; Binarized LS signs the fitted weights and retains the intercept; Direct Binary uses the sign of the class-mean difference with zero bias; STE Binary uses Adam with a straight-through binary weight and mean squared error on `{-1,+1}` targets, with learned bias; QUBO uses a signed-target squared-error objective and simulated annealing, with recovered bias. Validation selects thresholds; test is reserved for final metrics.

`stroke_binary_head_multiseed` writes a separate fixed-representation head-seed variability result set; it does not represent end-to-end BNN/DFA seed variability and must not be pooled with the controlled runner as the same experiment. Its Direct Binary variant greedily flips weight signs to improve BCE on `{0,1}` labels, unlike the controlled class-mean rule. Its STE variant uses mean BCE on `{0,1}` labels with Adam and learned bias, unlike the controlled signed-target MSE variant. Its QUBO variant uses signed-target squared error. Each output row records experiment type, variant, objective, target encoding, bias treatment, and optimization procedure. `stroke_real_ls` fits `{0,1}` targets; `stroke_dfa_least_squares` intentionally fits signed `{-1,+1}` targets with an intercept using `torch.linalg.lstsq`. The included tests cover split/provenance integrity, feature engineering, training-only fit, valid resampling, stale artifacts, all five heads, DFA BCE-gradient scaling, and rank-deficient least squares. ZIP CSVs and plots remain reference artifacts, not inputs to model runs.
