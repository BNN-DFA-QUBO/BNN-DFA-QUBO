BNN-DFA-QUBO

A research project investigating the combination of Binary Neural
Networks (BNNs), Direct Feedback Alignment (DFA), and a
QUBO/Ising-based binary classifier head on the MNIST dataset.

The repository contains a sequence of experiments that progressively
compare standard neural networks, BNNs trained with backpropagation,
BNNs trained using DFA, different binary classifier heads, and finally a
QUBO/Ising classifier optimized using simulated annealing.

1. Requirements

Software

Python 3.10--3.13

Git

pip

A virtual environment is recommended

Hardware

The project can run on a CPU.

A compatible GPU can optionally be used by PyTorch if available. The
QUBO/Ising optimization uses classical simulated annealing through
dwave-neal, so no D-Wave quantum computer, account, or API key is
required.

2. Clone the Repository

Clone the repository and enter its root directory:

git clone https://github.com/BNN-DFA-QUBO/BNN-DFA-QUBO.git
cd BNN-DFA-QUBO

Important: Keep the terminal in the repository root when running the
project commands.

The repository root should contain folders such as:

BNN-DFA-QUBO/
├── experiments/
├── models/
├── utils/
├── data/
├── README.md
└── requirements.txt

3. Create a Virtual Environment

Creating a virtual environment keeps the project's Python dependencies
separate from other projects.

Windows --- PowerShell

python -m venv venv
.\venv\Scripts\Activate.ps1

If PowerShell prevents activation because of its execution policy, you
can instead use Command Prompt or configure the execution policy for
your user account.

Windows --- Command Prompt

python -m venv venv
venv\Scripts\activate.bat

Linux / macOS

python3 -m venv venv
source venv/bin/activate

After activation, your terminal should indicate that the virtual
environment is active.

4. Install Dependencies

With the virtual environment activated:

python -m pip install --upgrade pip
pip install -r requirements.txt

The main dependencies include:

PyTorch

Torchvision

NumPy

SciPy

scikit-learn

pandas

matplotlib

dimod

dwave-neal

dwave-samplers

The exact versions are specified in requirements.txt.

5. Install the Project in Editable Mode

The project contains shared Python packages such as utils and
models. Install the repository itself in editable mode so that these
packages are registered with Python:

python -m pip install -e .

This command uses the repository's pyproject.toml.

Important: pip install -e . may create a package metadata
directory such as:

bnn_dfa_qubo.egg-info/

This is a generated build/install artifact and should not be
committed to Git. It should be covered by .gitignore.

The source directories themselves remain normal source code:

utils/
models/
experiments/

6. Python Package Structure and Imports

The repository uses utils and models as Python packages.

The expected structure is:

BNN-DFA-QUBO/
│
├── experiments/
│   ├── __init__.py
│   ├── test_data.py
│   ├── baseline_ann.py
│   ├── bnn_bp.py
│   ├── bnn_dfa.py
│   └── ...
│
├── models/
│   ├── __init__.py
│   ├── bnn.py
│   ├── dfa.py
│   └── qubo_head.py
│
├── utils/
│   ├── __init__.py
│   ├── data.py
│   └── seed.py
│
├── data/
├── pyproject.toml
├── requirements.txt
└── README.md

The __init__.py files can be empty. They make the package structure
explicit and keep imports predictable.

For example, imports can remain:

from utils.data import get_mnist_loaders

and:

from models.bnn import ...
from models.dfa import ...

Run experiments as Python modules

Use module execution for all experiment scripts:

python -m experiments.test_data
python -m experiments.baseline_ann
python -m experiments.bnn_bp

Do not use:

python experiments/test_data.py

or:

python experiments/baseline_ann.py

Module execution is important because it makes Python resolve the
project packages from the repository root consistently.

7. Dataset

The project uses the MNIST handwritten digit dataset:

60,000 training images

10,000 test images

28 × 28 grayscale images

10 classes (digits 0--9)

The dataset is handled by torchvision.

Automatic Download

Manual dataset downloading is normally not required.

The project uses:

datasets.MNIST(
    root="./data",
    download=True,
    ...
)

Therefore, when an experiment requiring MNIST is run for the first time,
Torchvision will automatically download the dataset.

The dataset will be stored relative to the repository root under:

data/
└── MNIST/
    └── raw/

The data/ directory is intended to be generated locally and is not
part of the source code that needs to be committed.

Recommended approach

After installing the dependencies and the project, first run:

python -m experiments.test_data

This verifies that the dataset can be downloaded and loaded correctly.

8. Manual Dataset Download

If the automatic MNIST download fails because of a network restriction,
proxy, firewall, or unavailable mirror, the four MNIST archive files can
be downloaded manually from the official MNIST source:

train-images-idx3-ubyte.gz

train-labels-idx1-ubyte.gz

t10k-images-idx3-ubyte.gz

t10k-labels-idx1-ubyte.gz

Place the files in:

BNN-DFA-QUBO/
└── data/
    └── MNIST/
        └── raw/
            ├── train-images-idx3-ubyte.gz
            ├── train-labels-idx1-ubyte.gz
            ├── t10k-images-idx3-ubyte.gz
            └── t10k-labels-idx1-ubyte.gz

The .gz files should be left compressed; Torchvision handles the
dataset processing.

9. Verify the Installation

Before running the experiments, verify that Python, the project
packages, PyTorch, Torchvision, and the dataset loader are working:

python -c "import utils; import models; print('Project imports: OK')"
python -m experiments.test_data

A successful dataset test should report a batch of MNIST images with a
shape similar to:

torch.Size([64, 1, 28, 28])

It should also report the number of training and test batches.

If this works, the environment, package imports, and dataset setup are
ready.

10. Running the Experiments

All experiment scripts are located in:

experiments/

Run them from the repository root and use Python module execution.

For example:

python -m experiments.baseline_ann

Do not run:

python experiments/baseline_ann.py

and do not:

cd experiments
python baseline_ann.py

Using module execution avoids common import errors involving packages
such as:

from models ...
from utils ...

Experiment 1 --- Standard ANN

python -m experiments.baseline_ann

A standard full-precision neural network trained using normal
backpropagation.

This provides the continuous baseline for comparison.

Experiment 2 --- BNN with Backpropagation

python -m experiments.bnn_bp

A Binary Neural Network trained using backpropagation and a
Straight-Through Estimator (STE).

Experiment 3 --- BNN with Direct Feedback Alignment

python -m experiments.bnn_dfa

A Binary Neural Network trained using Direct Feedback Alignment instead
of conventional backpropagation through the network.

Experiment 4 --- DFA + Least-Squares Head

python -m experiments.dfa_least_squares

Trains the BNN representation using DFA and then evaluates it using a
real-valued least-squares classifier head.

Experiment 5 --- DFA + Binary Least-Squares Head

python -m experiments.dfa_binary_head

Uses the least-squares solution as the basis for a binarized classifier
head.

Experiment 6A --- DFA + Tanh Binary Head

python -m experiments.dfa_direct_binary_head

Uses a continuous tanh relaxation to optimize a binary classifier
head.

Experiment 6B --- DFA + STE Binary Head

python -m experiments.dfa_binary_head_ste

Uses a Straight-Through Estimator to optimize a hard binary classifier
head.

Experiment 7 --- DFA + QUBO Binary Head

python -m experiments.bnn_dfa_qubo

This is the main QUBO experiment.

It:

Trains the BNN using DFA.

Extracts the hidden representations.

Constructs the binary classification problem.

Builds the corresponding QUBO/Ising formulation.

Optimizes the binary classifier using classical simulated annealing.

Evaluates the resulting classifier.

The simulated annealing is performed locally using dwave-neal.

No quantum hardware or D-Wave API credentials are required.

11. Recommended Run Order

For reproducing the complete experimental progression, run:

python -m experiments.test_data
python -m experiments.baseline_ann
python -m experiments.bnn_bp
python -m experiments.bnn_dfa
python -m experiments.dfa_least_squares
python -m experiments.dfa_binary_head
python -m experiments.dfa_direct_binary_head
python -m experiments.dfa_binary_head_ste
python -m experiments.bnn_dfa_qubo

The scripts should be treated as separate experiments unless the code
indicates otherwise. Running test_data first is recommended because it
confirms that the environment, package imports, and dataset are working.

12. Repository Structure

BNN-DFA-QUBO/
│
├── experiments/
│   ├── __init__.py
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
│   ├── __init__.py
│   ├── bnn.py
│   ├── dfa.py
│   └── qubo_head.py
│
├── utils/
│   ├── __init__.py
│   ├── data.py
│   └── seed.py
│
├── data/
│   └── MNIST/
│       └── raw/
│
├── pyproject.toml
├── .gitignore
├── requirements.txt
└── README.md

Main directories

experiments/

Contains the executable experiment modules.

models/

Contains the BNN, DFA, and QUBO-related model implementations.

utils/

Contains shared utilities such as MNIST loading and random seed
configuration.

data/

Local dataset storage. This is generated/downloaded locally and should
not need to be committed to Git.

13. Git / Generated Files

The following are generated locally and should normally remain
untracked:

venv/
data/
__pycache__/
*.pyc
*.egg-info/
build/
dist/

In particular, after:

python -m pip install -e .

you may see a directory ending in:

.egg-info/

This is generated package metadata and should be ignored by Git.

Do not ignore the actual source packages:

utils/
models/
experiments/

Those directories contain source code and should be committed.

Generated datasets and experiment outputs should also remain untracked
unless there is a specific reason to version them.

14. Troubleshooting

ModuleNotFoundError: No module named 'utils'

First make sure you are in the repository root:

cd BNN-DFA-QUBO

Then make sure the project has been installed in editable mode:

python -m pip install -e .

Then run the experiment as a module:

python -m experiments.test_data

Also verify:

python -c "import utils; import models; print('Project imports: OK')"

Do not run:

python experiments/test_data.py

or:

cd experiments
python test_data.py

ModuleNotFoundError: No module named 'models'

Use the same procedure:

cd BNN-DFA-QUBO
python -m pip install -e .
python -m experiments.bnn_dfa_qubo

Do not execute the experiment by directly passing its .py filepath.

MNIST Download Fails

If Torchvision cannot download MNIST:

Check your internet connection.

Try running the command again.

If your network blocks the download, manually download the MNIST
files and place them in:

data/MNIST/raw/

See the Dataset section above.

Dependency / Import Errors

Make sure the virtual environment is activated.

Windows

.\venv\Scripts\Activate.ps1

Linux/macOS

source venv/bin/activate

Then reinstall the dependencies:

python -m pip install -r requirements.txt
python -m pip install -e .

GPU Issues

The experiments can run on CPU. If PyTorch cannot access your GPU, the
project can still be run using the CPU where supported by the scripts.

The QUBO simulated annealing stage is performed using the CPU.

15. For Team Members

Every team member should independently perform:

git clone https://github.com/BNN-DFA-QUBO/BNN-DFA-QUBO.git
cd BNN-DFA-QUBO
python -m venv venv

Activate the environment and install:

python -m pip install -r requirements.txt
python -m pip install -e .

Then verify:

python -c "import utils; import models; print('Project imports: OK')"
python -m experiments.test_data

Once the test succeeds, the repository is ready to run.

Important Git note

Do not commit:

venv/

downloaded MNIST data

generated datasets

__pycache__/

*.egg-info/

build/

dist/

experiment outputs/checkpoints that are covered by .gitignore

Only commit source code, configuration, and documentation changes that
are intended to be shared with the rest of the team.

16. Quick Start

For someone who just wants to get started:

git clone https://github.com/BNN-DFA-QUBO/BNN-DFA-QUBO.git
cd BNN-DFA-QUBO
python -m venv venv

Windows PowerShell

.\venv\Scripts\Activate.ps1

Linux/macOS

source venv/bin/activate

Then:

python -m pip install -r requirements.txt
python -m pip install -e .
python -c "import utils; import models; print('Project imports: OK')"
python -m experiments.test_data

If the dataset test succeeds, run an experiment:

python -m experiments.baseline_ann

or run the main QUBO experiment:

python -m experiments.bnn_dfa_qubo

That's it. The repository is ready to use.