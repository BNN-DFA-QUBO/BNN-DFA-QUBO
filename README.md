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

The stroke pipeline downloads the public Fedesoriano Kaggle dataset locally. A raw CSV already present at `data/stroke/healthcare-dataset-stroke-data.csv` is preserved by default.

Run these commands from the repository root after installing `requirements.txt`:

```powershell
# Download; refuses to replace an existing local CSV.
python -m experiments.stroke_experiments.download_stroke
# To replace an existing local CSV intentionally, use:
# python -m experiments.stroke_experiments.download_stroke --overwrite

# Create deterministic stratified 70/15/15 splits, fit preprocessing, and balance train only.
python -m experiments.stroke_experiments.process_stroke --seed 42

# BNN trained with backpropagation.
python -m experiments.stroke_experiments.stroke_bnn_bp

# DFA representation, followed by all five DFA heads and a result summary.
python -m experiments.stroke_experiments.run_stroke_controlled

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
```

The data directory is ignored by Git. Set `STROKE_DATA_DIR` to choose another local data/artifact directory, or `STROKE_CSV` to point at an existing raw CSV. `STROKE_SPLIT_SEED` controls the default split seed; `process_stroke --seed` selects the processing seed. The processor writes `metadata.json` (dataset hash, row IDs per split, feature order and dynamic class counts) and `preprocessor.joblib` under `STROKE_DATA_DIR`. Each training call recreates deterministic splits and fits every transform on training rows only, then applies the fitted transforms to validation and test.

Feature engineering retains the ZIP choices: grouped work types, cardiovascular comorbidity, age/glucose interaction, glucose risk tier, metabolic syndrome flag, KNN imputation, Yeo–Johnson transforms, and one-hot categorical inputs. Input width is validated from the saved feature list (23 for the supplied schema). The ZIP's ordinary SMOTE plus rounding can create impossible multi-hot category groups, so this implementation balances only training data through exact minority-row duplication. That preserves binary and one-hot semantics and records original and balanced counts plus the method in metadata. Validation and test are untouched.

The DFA heads consume labels saved alongside their representations, avoiding independently regenerated split labels. Validation selects thresholds; test is used only for final reported metrics. The ZIP's CSVs and plots are reference artifacts and remain outside tracked source files.
