# BNN-DFA-QUBO

A research project investigating the combination of **Binary Neural Networks (BNNs)**, **Direct Feedback Alignment (DFA)**, and a **QUBO/Ising-based binary classifier head** on the MNIST dataset.

The repository contains a sequence of experiments that progressively compare standard neural networks, BNNs trained with backpropagation, BNNs trained using DFA, different binary classifier heads, and finally a QUBO/Ising classifier optimized using simulated annealing.

---

## 1. Requirements

### Software

* Python **3.10–3.13**
* Git
* `pip`
* A virtual environment is recommended

### Hardware

The project can run on a CPU.

A compatible GPU can optionally be used by PyTorch if available. The QUBO/Ising optimization uses **classical simulated annealing** through `dwave-neal`, so **no D-Wave quantum computer, account, or API key is required**.

---

# 2. Clone the Repository

Clone the repository and enter its root directory:

```bash
git clone https://github.com/BNN-DFA-QUBO/BNN-DFA-QUBO.git
cd BNN-DFA-QUBO
```

> **Important:** Keep the terminal in the repository root when running the experiment scripts. Do not `cd` into the `experiments/` directory.

The repository root should contain folders such as:

```text
BNN-DFA-QUBO/
├── experiments/
├── models/
├── utils/
├── data/
├── README.md
└── requirements.txt
```

---

# 3. Create a Virtual Environment

Creating a virtual environment keeps the project's Python dependencies separate from other projects.

### Windows — PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell prevents activation because of its execution policy, you can instead use Command Prompt or configure the execution policy for your user account.

### Windows — Command Prompt

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

After activation, your terminal should indicate that the virtual environment is active.

---

# 4. Install Dependencies

With the virtual environment activated:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The main dependencies include:

* PyTorch
* Torchvision
* NumPy
* SciPy
* scikit-learn
* pandas
* matplotlib
* `dimod`
* `dwave-neal`
* `dwave-samplers`

The exact versions are specified in `requirements.txt`.

---

# 5. Dataset

The project uses the **MNIST handwritten digit dataset**:

* 60,000 training images
* 10,000 test images
* 28 × 28 grayscale images
* 10 classes (digits 0–9)

The dataset is handled by `torchvision`.

## Automatic Download

**Manual dataset downloading is normally not required.**

The project uses:

```python
datasets.MNIST(
    root="./data",
    download=True,
    ...
)
```

Therefore, when an experiment requiring MNIST is run for the first time, Torchvision will automatically download the dataset.

The dataset will be stored relative to the repository root under:

```text
data/
└── MNIST/
    └── raw/
```

The `data/` directory is intended to be generated locally and is not part of the source code that needs to be committed.

### Recommended approach

After installing the dependencies, first run:

```bash
python experiments/test_data.py
```

This verifies that the dataset can be downloaded and loaded correctly.

---

## Manual Dataset Download

If the automatic MNIST download fails because of a network restriction, proxy, firewall, or unavailable mirror, the four MNIST archive files can be downloaded manually from the official MNIST source:

* `train-images-idx3-ubyte.gz`
* `train-labels-idx1-ubyte.gz`
* `t10k-images-idx3-ubyte.gz`
* `t10k-labels-idx1-ubyte.gz`

Place the files in:

```text
BNN-DFA-QUBO/
└── data/
    └── MNIST/
        └── raw/
            ├── train-images-idx3-ubyte.gz
            ├── train-labels-idx1-ubyte.gz
            ├── t10k-images-idx3-ubyte.gz
            └── t10k-labels-idx1-ubyte.gz
```

The `.gz` files should be left compressed; Torchvision handles the dataset processing.

---

# 6. Verify the Installation

Before running the experiments, verify that Python, PyTorch, Torchvision, and the dataset loader are working:

```bash
python experiments/test_data.py
```

A successful run should report a batch of MNIST images with a shape similar to:

```text
torch.Size([64, 1, 28, 28])
```

It should also report the number of training and test batches.

If this works, the environment and dataset setup are ready.

---

# 7. Running the Experiments

All experiment scripts are located in:

```text
experiments/
```

**Run them from the repository root.**

For example:

```bash
python experiments/baseline_ann.py
```

Do **not** do:

```bash
cd experiments
python baseline_ann.py
```

Running from the wrong directory can cause imports such as:

```python
from models...
from utils...
```

to fail.

---

## Experiment 1 — Standard ANN

```bash
python experiments/baseline_ann.py
```

A standard full-precision neural network trained using normal backpropagation.

This provides the continuous baseline for comparison.

---

## Experiment 2 — BNN with Backpropagation

```bash
python experiments/bnn_bp.py
```

A Binary Neural Network trained using backpropagation and a Straight-Through Estimator (STE).

---

## Experiment 3 — BNN with Direct Feedback Alignment

```bash
python experiments/bnn_dfa.py
```

A Binary Neural Network trained using Direct Feedback Alignment instead of conventional backpropagation through the network.

---

## Experiment 4 — DFA + Least-Squares Head

```bash
python experiments/dfa_least_squares.py
```

Trains the BNN representation using DFA and then evaluates it using a real-valued least-squares classifier head.

---

## Experiment 5 — DFA + Binary Least-Squares Head

```bash
python experiments/dfa_binary_head.py
```

Uses the least-squares solution as the basis for a binarized classifier head.

---

## Experiment 6A — DFA + Tanh Binary Head

```bash
python experiments/dfa_direct_binary_head.py
```

Uses a continuous `tanh` relaxation to optimize a binary classifier head.

---

## Experiment 6B — DFA + STE Binary Head

```bash
python experiments/dfa_binary_head_ste.py
```

Uses a Straight-Through Estimator to optimize a hard binary classifier head.

---

## Experiment 7 — DFA + QUBO Binary Head

```bash
python experiments/bnn_dfa_qubo.py
```

This is the main QUBO experiment.

It:

1. Trains the BNN using DFA.
2. Extracts the hidden representations.
3. Constructs the binary classification problem.
4. Builds the corresponding QUBO/Ising formulation.
5. Optimizes the binary classifier using classical simulated annealing.
6. Evaluates the resulting classifier.

The simulated annealing is performed locally using `dwave-neal`.

**No quantum hardware or D-Wave API credentials are required.**

---

# 8. Recommended Run Order

For reproducing the complete experimental progression, run:

```bash
python experiments/test_data.py
python experiments/baseline_ann.py
python experiments/bnn_bp.py
python experiments/bnn_dfa.py
python experiments/dfa_least_squares.py
python experiments/dfa_binary_head.py
python experiments/dfa_direct_binary_head.py
python experiments/dfa_binary_head_ste.py
python experiments/bnn_dfa_qubo.py
```

However, the scripts should be treated as separate experiments unless the code indicates otherwise. Running `test_data.py` first is recommended because it confirms that the environment and dataset are working.

---

# 9. Repository Structure

```text
BNN-DFA-QUBO/
│
├── experiments/
│   ├── test_data.py
│   ├── baseline_ann.py
│   ├── bnn_bp.py
│   ├── bnn_dfa.py
│   ├── dfa_least_squares.py
│   ├── dfa_binary_head.py
│   ├── dfa_direct_binary_head.py
│   ├── dfa_binary_head_ste.py
│   └── bnn_dfa_qubo.py
│
├── models/
│   ├── bnn.py
│   ├── dfa.py
│   └── qubo_head.py
│
├── utils/
│   ├── data.py
│   └── seed.py
│
├── data/
│   └── MNIST/
│       └── raw/
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Main directories

**`experiments/`**
Contains the executable experiment scripts.

**`models/`**
Contains the BNN, DFA, and QUBO-related model implementations.

**`utils/`**
Contains shared utilities such as MNIST loading and random seed configuration.

**`data/`**
Local dataset storage. This is generated/downloaded locally and should not need to be committed to Git.

---

# 10. Troubleshooting

## `ModuleNotFoundError: No module named 'models'`

Make sure you are running the command from the repository root.

Correct:

```bash
cd BNN-DFA-QUBO
python experiments/bnn_dfa_qubo.py
```

Incorrect:

```bash
cd BNN-DFA-QUBO/experiments
python bnn_dfa_qubo.py
```

---

## MNIST Download Fails

If Torchvision cannot download MNIST:

1. Check your internet connection.
2. Try running the command again.
3. If your network blocks the download, manually download the MNIST files and place them in:

```text
data/MNIST/raw/
```

See the **Dataset** section above.

---

## Dependency / Import Errors

Make sure the virtual environment is activated:

### Windows

```powershell
.\venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
source venv/bin/activate
```

Then reinstall the dependencies:

```bash
pip install -r requirements.txt
```

---

## GPU Issues

The experiments can run on CPU. If PyTorch cannot access your GPU, the project can still be run using the CPU where supported by the scripts.

The QUBO simulated annealing stage is performed using the CPU.

---

# 11. For Team Members

Every team member should independently perform:

```bash
git clone https://github.com/BNN-DFA-QUBO/BNN-DFA-QUBO.git
cd BNN-DFA-QUBO

python -m venv venv
```

Activate the environment and install:

```bash
pip install -r requirements.txt
```

Then verify:

```bash
python experiments/test_data.py
```

Once the test succeeds, the repository is ready to run.

### Important Git note

Do **not** commit:

* `venv/`
* downloaded MNIST data
* generated datasets
* experiment outputs/checkpoints that are covered by `.gitignore`

Only commit source-code, configuration, and documentation changes that are intended to be shared with the rest of the team.

---

# 12. Quick Start

For someone who just wants to get started:

```bash
git clone https://github.com/BNN-DFA-QUBO/BNN-DFA-QUBO.git
cd BNN-DFA-QUBO

python -m venv venv
```

### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
source venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
python experiments/test_data.py
```

If the dataset test succeeds, run an experiment:

```bash
python experiments/baseline_ann.py
```

or run the main QUBO experiment:

```bash
python experiments/bnn_dfa_qubo.py
```

That's it. The repository is ready to use.
