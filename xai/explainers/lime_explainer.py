"""Tabular LIME explainer.

Uses the official ``lime`` package when installed. A transparent in-project
weighted local linear surrogate is provided as a dependency-free fallback;
outputs record which backend actually ran.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def _internal_lime(predict_scalar, training_data, instance, *, num_samples, seed, feature_names):
    rng = np.random.default_rng(int(seed))
    train = np.asarray(training_data, dtype=np.float64)
    x = np.asarray(instance, dtype=np.float64).reshape(-1)
    std = np.std(train, axis=0, ddof=0)
    std = np.where(std > 1e-8, std, 1.0)
    # LIME-style local perturbations: normal draws in standardized space,
    # inverted back to the model's original feature space for prediction.
    z = rng.normal(0.0, 1.0, size=(int(num_samples), x.size))
    z = np.vstack([np.zeros((1, x.size)), z])
    perturbed = x[None, :] + z * std[None, :]
    # Match LIME's tabular neighborhood scaling: Euclidean distance in
    # standardized feature space and the default 0.75 * sqrt(n_features)
    # kernel width. Using RMS distance here would make high-dimensional
    # neighborhoods almost uniformly weighted and insufficiently local.
    distances = np.linalg.norm(z, axis=1)
    kernel_width = max(np.sqrt(x.size) * 0.75, 1e-6)
    weights = np.sqrt(np.exp(-(distances ** 2) / (kernel_width ** 2)))
    y = np.asarray(predict_scalar(perturbed), dtype=np.float64).reshape(-1)
    if y.size != perturbed.shape[0] or not np.isfinite(y).all():
        raise ValueError("Predictor returned invalid outputs during LIME perturbation")
    # Fit on standardized perturbations so feature scales do not distort the
    # local surrogate regularization. Coefficients are per one training SD.
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(z, y, sample_weight=weights)
    fitted = model.predict(z)
    local_r2 = float(r2_score(y, fitted, sample_weight=weights))
    local_prediction = float(model.predict(np.zeros((1, x.size)))[0])
    # Match the official LIME adapter's semantics:
    # - weights: local coefficient per one training standard deviation
    # - values: per-instance local contribution (weight * standardized offset)
    # This keeps SHAP/LIME agreement and faithfulness rankings comparable
    # regardless of which backend is installed.
    mean = np.mean(train, axis=0)
    contributions = np.asarray(model.coef_, dtype=np.float64) * ((x - mean) / std)
    return {
        "values": contributions,
        "weights": np.asarray(model.coef_, dtype=np.float64),
        "intercept": float(model.intercept_),
        "local_prediction": local_prediction,
        "instance_prediction": float(y[0]),
        "local_r2": local_r2,
        "backend": "internal_lime_weighted_ridge",
        "num_samples": int(num_samples),
    }


def explain_one(predict_scalar, training_data, instance, *, num_samples=2000, seed=42, feature_names=None):
    train = np.asarray(training_data, dtype=np.float64)
    x = np.asarray(instance, dtype=np.float64).reshape(-1)
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return _internal_lime(predict_scalar, train, x, num_samples=num_samples, seed=seed, feature_names=feature_names)
    explainer = LimeTabularExplainer(
        train, feature_names=feature_names, mode="regression",
        discretize_continuous=False, random_state=int(seed),
    )
    exp = explainer.explain_instance(
        x, lambda batch: np.asarray(predict_scalar(batch), dtype=np.float64).reshape(-1),
        num_features=x.size, num_samples=int(num_samples),
    )
    weights = np.zeros(x.size, dtype=np.float64)
    for idx, coef in exp.as_map().get(0, []):
        weights[int(idx)] = float(coef)
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0, ddof=0)
    std = np.where(std > 1e-8, std, 1.0)
    contributions = weights * ((x - mean) / std)
    local_prediction = float(np.asarray(exp.local_pred).reshape(-1)[0])
    return {
        "values": contributions,
        "weights": weights,
        "intercept": float(np.asarray(exp.intercept[0] if isinstance(exp.intercept, dict) else exp.intercept).reshape(-1)[0]),
        "local_prediction": local_prediction,
        "instance_prediction": float(np.asarray(predict_scalar(x[None, :])).reshape(-1)[0]),
        "local_r2": float(exp.score),
        "backend": "lime.lime_tabular.LimeTabularExplainer",
        "num_samples": int(num_samples),
    }
