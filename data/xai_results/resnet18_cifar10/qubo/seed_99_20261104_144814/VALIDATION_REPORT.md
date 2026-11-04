# XAI experiment report: resnet18_cifar10 / qubo

- Seed: 99
- Explanation scope: classifier head over frozen ResNet18 512-dimensional penultimate features
- Representation SHA-256: `36751063dced722c2bbc15e71570382f605180b3aaada8d66d6b51bb9663c529`
- Test accuracy: 0.9508
- Test balanced accuracy: 0.9508
- Test macro F1: 0.9508

## Explanation validation

- SHAP local-accuracy pass rate (absolute error <= 0.03): 1.000
- Mean LIME weighted local R²: 1.0000
- Mean SHAP/LIME absolute-importance rank correlation: 0.9650
- Mean SHAP stability rank correlation (independent background samples): 0.9380
- Mean LIME stability rank correlation (independent perturbation seeds): 0.9991

## Files

- `feature_attributions.csv`: per-sample, per-feature SHAP and LIME values.
- `sample_metrics.csv`: explanation diagnostics and perturbation faithfulness metrics.
- `summary_metrics.csv`: numeric summary.
- `run_manifest.json`: configuration and provenance.

## Important interpretation limits

Ablation metrics measure sensitivity to the selected training-median replacement and are not causal effects. Latent features are not directly equivalent to original input features.
These are pilot explanations on held-out test rows. They do not establish causal effects or population-level scientific conclusions. Multi-seed runs and prespecified analysis are required for study claims.
