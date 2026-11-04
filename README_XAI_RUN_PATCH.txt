XAI RUNNER + RESULT AGGREGATION PATCH
====================================

This is an additive patch. It adds two scripts under xai/ and does not overwrite
or alter model, training, QUBO, SHAP, LIME, or representation code.

Apply from the repository root:
    unzip -o ~/Downloads/xai_runner_aggregation_patch.zip -d .

Install dependencies in the active project venv:
    python -m pip install -r xai/requirements.txt
    python -m pip install dwave-neal

The full five-head XAI matrix for one dataset and one seed:
    python -m xai.run_all_heads --datasets bank --seed 42

Recommended stronger protocol for all four datasets, five heads, and five seeds:
    python -m xai.run_all_heads --datasets bank mnist cifar10 stroke --seeds 42 123 2024 7 99 --explain-samples 20 --background-size 40 --lime-samples 3000 --stability-repeats 5 --num-reads 100 --num-sweeps 1000

IMPORTANT PREREQUISITES:
- Bank files/preprocessing are expected in the repository.
- MNIST/CIFAR-10 data may download through the existing torchvision data utilities.
- Stroke requires the raw dataset at the location expected by the existing
  experiments/stroke_experiments/download_stroke.py pipeline.
- Actual QUBO runs require dwave-neal (`neal`). The runner checks it before
  starting and fails instead of silently substituting another classifier.
- Official LIME is required by default. If you explicitly pass
  --allow-lime-fallback, the existing internal weighted-Ridge fallback may run;
  its output must be labelled as a fallback, not official LIME.
- The run-all command is intentionally fail-fast. If a dataset or solver is
  unavailable, it stops and reports the failed run. Run datasets separately
  when prerequisites differ.
- This XAI implementation explains classifier heads over latent frozen BNN-DFA
  representations, not raw pixels or original input columns.
- Do not interpret median-replacement ablation as causal evidence.

After the requested runs finish, aggregate:
    python -m xai.collect_xai_results --datasets bank mnist cifar10 stroke --seeds 42 123 2024 7 99

Outputs:
    data/xai_results/aggregated/xai_metrics_by_seed.csv
    data/xai_results/aggregated/xai_metrics_mean_sd.csv

The collector averages explanation-level metrics within each run first, then
calculates mean and sample standard deviation across independent seeds. It does
not incorrectly treat multiple explained rows from one seed as independent
experimental replicates. Missing runs are reported, never imputed.
