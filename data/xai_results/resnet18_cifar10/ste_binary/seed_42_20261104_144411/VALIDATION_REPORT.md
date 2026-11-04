# XAI experiment report: resnet18_cifar10 / ste_binary

- Seed: 42
- Explanation scope: classifier head over frozen ResNet18 512-dimensional penultimate features
- Representation SHA-256: `abca9964dcc9782183ba95dfdde56bf15ccbb666adf6abb4df915bbf777a8e06`
- Test accuracy: 0.9470
- Test balanced accuracy: 0.9470
- Test macro F1: 0.9470

## Explanation validation

- SHAP local-accuracy pass rate (absolute error <= 0.03): 1.000
- Mean LIME weighted local R²: 1.0000
- Mean SHAP/LIME absolute-importance rank correlation: 0.8520
- Mean SHAP stability rank correlation (independent background samples): 0.8472
- Mean LIME stability rank correlation (independent perturbation seeds): 0.9127

## Files

- `feature_attributions.csv`: per-sample, per-feature SHAP and LIME values.
- `sample_metrics.csv`: explanation diagnostics and perturbation faithfulness metrics.
- `summary_metrics.csv`: numeric summary.
- `run_manifest.json`: configuration and provenance.

## Important interpretation limits

Ablation metrics measure sensitivity to the selected training-median replacement and are not causal effects. Latent features are not directly equivalent to original input features.
These are pilot explanations on held-out test rows. They do not establish causal effects or population-level scientific conclusions. Multi-seed runs and prespecified analysis are required for study claims.
