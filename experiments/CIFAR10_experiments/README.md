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

## Experimental Tracks: Baseline vs. opt_control Stabilization

The CIFAR-10 investigation includes two distinct experimental configurations that address separate research questions. **These are separate experimental configurations and their results must not be conflated.**

| Configuration Parameter | Original Baseline (`experiments/CIFAR10_experiments/`) | Stabilized `opt_control` (`experiments/CIFAR10_experiments/stabilization/`) |
|:---|:---|:---|
| **Learning Rate** | `0.001` ($10^{-3}$) | `0.0001` ($10^{-4}$) |
| **Gradient Clipping** | None (unclipped) | `1.0` (max norm clipping) |
| **Weight Decay** | None ($0.0$) | `0.0001` ($10^{-4}$, on latent weights) |
| **Loss & Activation Behavior** | Activation norm grows ($H_{\text{norm}} \to 4,335.2$), loss inflates | Bounded activation norm ($H_{\text{norm}} \approx 109.1$), loss stable ($\approx 2.41$) |
| **Artifact Directory** | `data/cifar10_representations/` | `data/cifar10_stabilization/representations/` |
| **Results Log** | `data/results/cifar10_results.csv` | `data/cifar10_stabilization/results/` |
| **Purpose** | Evaluates original unregularized BNN-DFA dynamics on CIFAR-10 | Evaluates bounded BNN-DFA dynamics and tests stabilization regimes |

### Conflation Warning
- **Do not mix representations:** Baseline representations and stabilized representations are stored separately and must never be swapped across evaluation tracks.
- **Do not confound head modifications with backbone stabilization:** The Direct Binary comparison between original unscaled baseline ($10.13\%$) and stabilized scale/bias-refitted head ($13.19\%$) is an end-to-end pipeline comparison. On identical stabilized representations, the isolated head change is $+1.68\%$ ($p = 0.1540$, not statistically significant).

## QUBO Validation Tuning Diagnostics

Whenever validation-based QUBO grid tuning is enabled (`--qubo-tuning` or `fixed_qubo=False`), candidate-level diagnostic records are logged via `write_tuning_csv()` to:
```text
data/results_diagnostic/cifar10_qubo/
```

Logged fields include:
- Experiment metadata: seed, backbone training epochs, validation fraction, representation artifact ID, sample counts ($N_{\text{train}}, N_{\text{val}}, N_{\text{test}}$).
- Grid parameters: `num_reads`, `num_sweeps`, `scale_multiplier`, `solver_seed`.
- Candidate evaluation: `validation_accuracy`, `selected` flag, `runtime_sec`, `tuning_total_seconds`.
- Health status: `status` (`ok` vs. `failed`) and `error` string.

> **Methodological Rule:** Test accuracy is strictly excluded from candidate tuning CSVs and selection routines to guarantee zero test-set leakage.

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
