"""Numerical, local-fidelity, agreement, stability and perturbation metrics."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


def rank_correlation(a, b) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else float("nan")


def topk_jaccard(a, b, k: int) -> float:
    a = set(np.argsort(np.abs(np.asarray(a))) [-k:])
    b = set(np.argsort(np.abs(np.asarray(b))) [-k:])
    return float(len(a & b) / max(len(a | b), 1))


def faithfulness_metrics(predict_scalar, x, attribution, background, *, top_k=10, repeats=20, seed=42):
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=float).reshape(-1)
    attr = np.asarray(attribution, dtype=float).reshape(-1)
    background = np.asarray(background, dtype=float)
    k = min(int(top_k), x.size)
    baseline = np.median(background, axis=0)
    original = float(np.asarray(predict_scalar(x[None, :])).reshape(-1)[0])
    top_idx = np.argsort(np.abs(attr))[-k:]
    masked = x.copy(); masked[top_idx] = baseline[top_idx]
    top_impact = abs(original - float(np.asarray(predict_scalar(masked[None, :])).reshape(-1)[0]))
    random_impacts = []
    for _ in range(int(repeats)):
        idx = rng.choice(x.size, size=k, replace=False)
        row = x.copy(); row[idx] = baseline[idx]
        score = float(np.asarray(predict_scalar(row[None, :])).reshape(-1)[0])
        random_impacts.append(abs(original - score))
    random_mean = float(np.mean(random_impacts))
    return {
        "topk_ablation_abs_change": float(top_impact),
        "random_ablation_abs_change_mean": random_mean,
        "random_ablation_abs_change_std": float(np.std(random_impacts, ddof=1)) if len(random_impacts) > 1 else 0.0,
        "topk_vs_random_ratio": float(top_impact / random_mean) if random_mean > 1e-12 else float("nan"),
        "top_k": int(k),
        "ablation_baseline": "training_background_feature_median",
    }
