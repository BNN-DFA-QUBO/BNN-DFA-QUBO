"""Model-agnostic Kernel SHAP wrapper with stable output normalization."""
from __future__ import annotations
import numpy as np


def _as_single_output(values, n_samples: int, n_features: int) -> np.ndarray:
    # SHAP versions differ: list[class] or ndarray[sample, feature, output].
    if isinstance(values, list):
        values = values[0]
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 3:
        # This function is called with a scalar-output model; tolerate a trailing singleton only.
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1 and arr.shape[1:] == (n_samples, n_features):
            arr = arr[0]
        else:
            raise ValueError(f"Unexpected multi-output SHAP shape {arr.shape}")
    if arr.shape != (n_samples, n_features):
        raise ValueError(f"SHAP values have shape {arr.shape}; expected {(n_samples, n_features)}")
    if not np.isfinite(arr).all():
        raise ValueError("SHAP returned NaN or infinite attributions")
    return arr


def explain_kernel(predict_scalar, background, samples, *, nsamples: int, link: str = "identity"):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required. Install xai/requirements.txt in the project environment.") from exc
    background = np.asarray(background, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    explainer = shap.KernelExplainer(predict_scalar, background, link=link)
    raw = explainer.shap_values(samples, nsamples=int(nsamples), silent=True, l1_reg=0)
    values = _as_single_output(raw, samples.shape[0], samples.shape[1])
    base = np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)
    if base.size != 1:
        raise ValueError(f"Expected scalar SHAP base value; got {base.shape}")
    predictions = np.asarray(predict_scalar(samples), dtype=np.float64).reshape(-1)
    additivity_error = np.abs(float(base[0]) + values.sum(axis=1) - predictions)
    return {
        "values": values,
        "base_value": float(base[0]),
        "predictions": predictions,
        "additivity_error": additivity_error,
        "backend": "shap.KernelExplainer",
        "nsamples": int(nsamples),
    }
