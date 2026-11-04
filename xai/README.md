# XAI: SHAP and LIME for BNN-DFA-QUBO

This package adds a separate explanation/evaluation layer to the existing BNN-DFA-QUBO research project. It does not replace or rewrite the existing training algorithms. It uses the repository's validated, shared frozen BNN-DFA representations and then fits the project's existing classifier-head procedures before explaining their scores.

## Scope (read this before interpreting results)

The current implementation explains the **classifier head over the frozen hidden representation**. For MNIST and CIFAR-10, this means latent activations rather than raw image pixels. For Bank and Stroke, it means latent activations rather than the raw preprocessed input columns. Results are therefore suitable for comparing classifier heads on a common representation; they are not pixel-attribution maps or direct explanations of original feature columns. The representation artifact SHA-256 is saved with every run.

Supported dataset/method experiment files:

| Dataset | Experiment files |
|---|---|
| MNIST | `xai/experiments/mnist/xai_{real_ls,binarized_ls,direct_binary,ste_binary,qubo}.py` |
| Stroke | `xai/experiments/stroke/xai_{real_ls,binarized_ls,direct_binary,ste_binary,qubo}.py` |
| Bank | `xai/experiments/bank/xai_{real_ls,binarized_ls,direct_binary,ste_binary,qubo}.py` |
| CIFAR-10 | `xai/experiments/cifar10/xai_{real_ls,binarized_ls,direct_binary,ste_binary,qubo}.py` |

The experiment files are intentionally separate and hard-coded. No interactive terminal model selector is used.

## Setup

From the repository root and with the project's active virtual environment:

```powershell
python -m pip install -r xai/requirements.txt
```

`shap` is used via `shap.KernelExplainer`. The LIME adapter prefers the official `lime.lime_tabular.LimeTabularExplainer`. When the `lime` package is unavailable, it uses the documented in-project weighted local-linear surrogate (`internal_lime_weighted_ridge`) so that the pipeline can still be tested. Each run records the actual backend. For research runs, install the official dependency and confirm `lime_backend` is `lime.lime_tabular.LimeTabularExplainer` in `sample_metrics.csv`.

The QUBO experiment files additionally require the project's `dwave-neal` / `neal` dependency already listed in the root `requirements.txt`.

## Commands

Run from the repository root. Examples:

```powershell
python -m xai.experiments.bank.xai_real_ls --seed 42 --epochs 15 --background-size 40 --explain-samples 8
python -m xai.experiments.bank.xai_binarized_ls --seed 42 --epochs 15
python -m xai.experiments.mnist.xai_real_ls --seed 42 --epochs 16
python -m xai.experiments.stroke.xai_real_ls --seed 42 --epochs 15
python -m xai.experiments.cifar10.xai_real_ls --seed 42 --epochs 20
```

Use `--epochs` to control the backbone representation training protocol. The dataset-specific default is used when omitted. MNIST/CIFAR data downloads follow the existing torchvision dataset utilities. Stroke requires the existing raw CSV under the location documented by `experiments/stroke_experiments/download_stroke.py`; the XAI code does not invent missing data. Bank uses the repository's committed dataset/preprocessing pipeline.

## Output layout

Each run writes to:

```
data/xai_results/<dataset>/<method>/seed_<seed>_<timestamp>/
    feature_attributions.csv
    sample_metrics.csv
    summary_metrics.csv
    run_manifest.json
    VALIDATION_REPORT.md
    shap_global_importance.png
    lime_global_importance.png
    shap_lime_global_comparison.png
```

A separate representation cache is written under `data/xai_artifacts/<dataset>/representations/`; existing experiment caches are not overwritten. A run manifest binds the results to the representation artifact SHA-256, head fit configuration, seeds, feature schema, data counts, and explanation backends.

## Methodology and checks

- SHAP: model-agnostic Kernel SHAP on the selected class-versus-strongest-competitor decision-score margin. This continuous target avoids saturated probabilities and uses a fixed competitor for each explained row. Default `nsamples = 2 * n_features + 2048` unless explicitly overridden.
- LIME: official `LimeTabularExplainer` when installed; otherwise a documented local weighted-Ridge implementation with LIME-style standardized perturbations, Euclidean kernel distance, and default kernel-width scaling. Per-feature coefficients (per training standard deviation) and per-instance local contributions are both retained; backend semantics are normalized so SHAP/LIME ranking diagnostics compare the same per-instance contribution quantity. Local R² and backend are recorded.
- Stability: each explanation is recomputed using an independent background sample (SHAP) or perturbation RNG seed (LIME); Spearman and top-k Jaccard compare absolute-importance rankings.
- SHAP local accuracy: check `base_value + sum(phi) ~= selected class-vs-competitor decision-score margin`, with a 0.03 absolute-error review threshold. The saved `shap_model_output` is this margin; `predicted_class_probability` is retained separately for classification context.
- Cross-method agreement: Spearman rank correlation between absolute SHAP per-instance contributions and absolute LIME per-instance local contributions. The global LIME bar plot uses coefficient magnitude per training standard deviation, a sensitivity scale; the comparison plot uses per-instance contributions so the two methods are compared on a consistent row-level basis.
- Perturbation faithfulness: replace the top-k attributed features with training-background medians and compare the absolute prediction change with repeated random-k feature ablations. This is a sensitivity diagnostic, not a causal estimate.
- Test labels are never used to fit heads, tune thresholds, or select models. The explanation subset is deterministically stratified by true test class to include both classes; labels are otherwise used only for post-hoc test metrics and row identification. The manifest records the sampling rule and exact test-row indices.

## Research-use cautions

A successful pilot run establishes that the code path executes and numerical checks are internally consistent; it does not establish publishable population-level evidence. A paper-level study should prespecify multiple seeds, sample counts, SHAP/LIME budgets, and evaluation thresholds; retain each seed's raw outputs; use official LIME; and report confidence intervals across seeds. Do not claim original-input explanations from latent-unit attributions. Ablation is distribution-dependent and not causal.

## Tests

From the project root:

```powershell
python -m pytest xai/tests -q
```

The tests include deterministic synthetic predictors, finite-value/shape checks, SHAP local-accuracy checks, LIME local fit checks, and faithfulness/stability metric tests. Real dataset runs are separate from unit tests.

## Coverage boundary

This first XAI release covers the five **classifier-head comparison** methods on the repository's shared frozen BNN-DFA representations. It does not claim to explain the separate end-to-end `baseline_ann`, `bnn_bp`, or `bnn_dfa` training runs. Those models have distinct learned representations and require a separate, method-specific extraction protocol; treating their features as though they were the shared head-comparison representation would be scientifically misleading. CIFAR-10's supported scope is the five heads on its shared convolutional BNN-DFA representation.

The included Bank validation outputs were generated in a sandbox where the official `lime` package was not installed. Their recorded backend is `internal_lime_weighted_ridge`, a transparent LIME-style fallback. The standard outputs are saved under `data/xai_results/`. For official LIME runs, install `xai/requirements.txt`, rerun the experiment, and verify the backend in `run_manifest.json` before using those LIME outputs as research evidence.
