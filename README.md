BNN-DFA-QUBO

A research project investigating the combination of Binary Neural
Networks (BNNs), Direct Feedback Alignment (DFA), and a
QUBO/Ising-based binary classifier head on the MNIST dataset.

The repository contains a sequence of experiments that progressively
compare standard neural networks, BNNs trained with backpropagation,
BNNs trained using DFA, different binary classifier heads, and finally a
QUBO/Ising classifier optimized using simulated annealing.

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

python -m experiments.test_data
python -m experiments.baseline_ann
python -m experiments.bnn_bp
python -m experiments.bnn_dfa
python -m experiments.dfa_least_squares
python -m experiments.dfa_binary_head
python -m experiments.dfa_direct_binary_head
python -m experiments.dfa_binary_head_ste
python -m experiments.bnn_dfa_qubo

Project structure

BNN-DFA-QUBO/
├── experiments/
├── models/
├── utils/
├── data/
├── pyproject.toml
├── requirements.txt
└── README.md