# DFA-BNN-QUBO

Research project investigating the combination of:

- Binary Neural Networks (BNNs)
- Direct Feedback Alignment (DFA)
- QUBO-based binary classifier optimization

The project evaluates different training and classifier-head approaches on the MNIST dataset.

---

## Project Objective

The main objective of this project is to study whether QUBO-based optimization can be used to obtain an effective binary classifier head for a neural network trained using Direct Feedback Alignment.

The experiments are organized as a progression:

1. Standard ANN + Backpropagation
2. BNN + Backpropagation
3. BNN + Direct Feedback Alignment
4. BNN + DFA + Real-Valued Least-Squares Head
5. BNN + DFA + Binary Classifier Head
6. BNN + DFA + STE Binary Classifier Head
7. BNN + DFA + QUBO Binary Classifier Head

The experiments are currently performed on MNIST.

---

## Repository Structure

```text
DFA-BNN-QUBO/
│
├── data/
│   └── # MNIST dataset (not tracked by Git)
│
├── experiments/
│   ├── baseline_ann.py
│   ├── bnn_bp.py
│   ├── bnn_dfa.py
│   ├── dfa_least_squares.py
│   ├── dfa_binary_head.py
│   ├── dfa_direct_binary_head.py
│   ├── dfa_binary_head_ste.py
│   ├── bnn_dfa_qubo.py
│   └── test_data.py
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
├── results/
│   └── # Generated results (not tracked by Git)
│
├── .gitignore
├── README.md
└── requirements.txt