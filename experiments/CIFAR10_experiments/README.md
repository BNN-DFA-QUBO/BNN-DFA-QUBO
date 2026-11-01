# CIFAR-10 Experiment Track: BNN + DFA + QUBO

This folder contains the complete CIFAR-10 experiment track investigating how different binary classifier-head optimization methods, particularly QUBO-based simulated annealing, affect the use of DFA-learned representations in Binary Neural Networks.

## Pipeline

```text
CIFAR-10 -> BNN -> DFA training -> learned 128-d representation -> classifier heads -> comparison
```

All five classifier heads are evaluated on the exact same frozen representation artifact for a given seed:

1. **Real Least Squares (Ridge LS)**: Regularized pseudo-inverse solving an augmented system with unpenalized intercept.
2. **Binarized Least Squares**: Hard bipolarization `{-1, +1}` with class-specific scale and bias refit on training representations.
3. **Direct Binary**: Continuous `tanh` relaxation optimized with Adam, followed by sign projection.
4. **STE Binary**: Straight-Through Estimator with validation-only checkpoint selection.
5. **QUBO Binary**: Ising Hamiltonian formulation with class-specific scales, solved via classical simulated annealing (`dwave-neal`), followed by post-annealing scale and bias fitting.

## Directory Structure

```text
experiments/CIFAR10_experiments/
├── __init__.py
├── README.md
├── cifar_bnn_dfa.py              # Backbone BNN + DFA training
├── cifar_representation.py        # Representation extraction & caching
├── cifar_dfa_least_squares.py     # Head 1: Real Ridge LS
├── cifar_dfa_binarized_ls.py      # Head 2: Binarized Ridge LS
├── cifar_dfa_direct_binary.py     # Head 3: Direct Binary
├── cifar_dfa_ste_binary.py        # Head 4: STE Binary
├── cifar_dfa_qubo.py              # Head 5: QUBO Binary
├── tune_qubo_validation.py        # QUBO grid tuning on validation set
├── run_cifar10_shared_validation.py # Canonical 5-head comparison on shared representation
├── run_cifar10_controlled.py      # Multi-seed controlled runner
├── results_utils.py               # Standardized logging to data/results/cifar10_results.csv
├── collect_results.py             # Aggregate table generator
├── plot_results.py                # Publication figure generation
└── stabilization/                 # Diagnostic & stabilization track
    ├── __init__.py
    ├── run_stabilization_experiment.py # 5-seed controlled stabilized runner (opt_control)
    ├── stabilized_direct_binary.py     # Stabilized Direct Binary with post-projection refit
    └── plot_stabilization_results.py   # Publication figures (Fig 3, 4, 5)
```

## Running Experiments

### 1. Run Baseline Shared Validation Comparison (One Seed)
```bash
python -m experiments.CIFAR10_experiments.run_cifar10_shared_validation --seed 42 --epochs 20
```

### 2. Run Baseline Controlled Multi-Seed Experiment
```bash
python -m experiments.CIFAR10_experiments.run_cifar10_controlled --seeds 42 123 456 789 999 --epochs 20
```

### 3. Run Stabilized Track (Controlled 5-Seed Benchmark & Diagnostic)
```bash
python -m experiments.CIFAR10_experiments.stabilization.run_stabilization_experiment --seeds 42 123 456 789 999 --epochs 20
python -m experiments.CIFAR10_experiments.stabilization.plot_stabilization_results
```

### 4. Collect Result Tables & Figures
```bash
python -m experiments.CIFAR10_experiments.collect_results
python -m experiments.CIFAR10_experiments.plot_results
```
