"""End-to-end SHAP/LIME experiment runner for shared BNN-DFA representations."""
from __future__ import annotations
import json
import math
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from xai.config import XAIConfig
from xai.core.contracts import validate_matrix, validate_predictor
from xai.explainers.shap_explainer import explain_kernel
from xai.explainers.lime_explainer import explain_one
from xai.evaluation.metrics import rank_correlation, topk_jaccard, faithfulness_metrics
from xai.evaluation.validation import validate_explanation, local_accuracy_status
from xai.integrations.representations import load_dataset_representation
from xai.integrations.heads import fit_head


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try: return value.item()
        except Exception: pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _select_indices(y, n, seed):
    """Deterministically sample across true classes without using labels for tuning."""
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, int(math.ceil(n / max(len(classes), 1))))
    chosen = []
    for cls in classes:
        rows = np.flatnonzero(y == cls)
        rng.shuffle(rows)
        chosen.extend(rows[:min(per_class, len(rows))].tolist())
    if len(chosen) < n:
        remaining = np.setdiff1d(np.arange(len(y)), np.asarray(chosen, dtype=int))
        rng.shuffle(remaining)
        chosen.extend(remaining[:n-len(chosen)].tolist())
    rng.shuffle(chosen)
    return np.asarray(chosen[:n], dtype=int)


def _sample_background(X, n, seed):
    rng = np.random.default_rng(seed)
    if len(X) <= n:
        return X.copy()
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx].copy()


def _decision_margin_predictor(head, predicted_class: int, reference_row: np.ndarray):
    """Return a scalar predictor for the selected class-vs-competitor score margin.

    Explaining probabilities can be numerically uninformative for saturated
    softmax/sigmoid outputs. A fixed class-vs-competitor margin is a continuous
    decision score and avoids changing the competitor class across perturbed
    rows, keeping SHAP additivity and LIME locality well-defined.
    """
    reference_scores = np.asarray(head.predict_scores(np.asarray(reference_row)[None, :]), dtype=np.float64)
    if reference_scores.ndim == 1:
        direction = 1.0 if int(predicted_class) == 1 else -1.0
        return lambda batch, sign=direction: sign * np.asarray(head.predict_scores(batch), dtype=np.float64).reshape(-1)
    if reference_scores.ndim != 2 or reference_scores.shape[0] != 1 or reference_scores.shape[1] < 2:
        raise ValueError(f"predict_scores must return [n, classes] or [n] for margin explanations; got {reference_scores.shape}")
    n_classes = reference_scores.shape[1]
    cls = int(predicted_class)
    if not 0 <= cls < n_classes:
        raise ValueError(f"predicted class {cls} is outside score width {n_classes}")
    competitors = [j for j in range(n_classes) if j != cls]
    competitor = max(competitors, key=lambda j: reference_scores[0, j])
    def predict_margin(batch, c=cls, other=competitor):
        scores = np.asarray(head.predict_scores(batch), dtype=np.float64)
        return scores[:, c] - scores[:, other]
    return predict_margin


def _mean_finite(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _plot_global(attributions: pd.DataFrame, output_dir: Path, feature_names: list[str]):
    if attributions.empty:
        return []
    files = []
    for col, title, filename in (("shap_value", "Mean absolute SHAP value", "shap_global_importance.png"),
                                 ("lime_weight_per_sd", "Mean absolute LIME coefficient (per SD)", "lime_global_importance.png")):
        grouped = attributions.groupby("feature_name")[col].apply(lambda s: float(np.mean(np.abs(s)))).sort_values().tail(20)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(grouped.index.astype(str), grouped.values)
        ax.set_xlabel("Mean absolute attribution")
        ax.set_title(title)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(str(path))
    # Paired comparison across per-sample standardized ranks (not raw magnitudes).
    fig, ax = plt.subplots(figsize=(7, 6))
    mean_shap = attributions.groupby("feature_name")["shap_value"].apply(lambda s: float(np.mean(np.abs(s))))
    mean_lime = attributions.groupby("feature_name")["lime_contribution"].apply(lambda s: float(np.mean(np.abs(s))))
    aligned = pd.concat([mean_shap.rename("SHAP"), mean_lime.rename("LIME")], axis=1).fillna(0)
    ax.scatter(aligned["SHAP"], aligned["LIME"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Mean |LIME local contribution|")
    ax.set_title("Global attribution magnitude comparison")
    fig.tight_layout()
    path = output_dir / "shap_lime_global_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    files.append(str(path))
    return files


def run_xai_experiment(dataset: str, method: str, config: XAIConfig | None = None) -> Path:
    config = config or XAIConfig()
    config.validate()
    dataset, method = dataset.lower(), method.lower()
    started = datetime.now(timezone.utc)
    rep = load_dataset_representation(
        dataset, seed=config.seed, epochs=config.backbone_epochs,
        device=config.device, artifact_dir=config.artifact_dir,
        overwrite=config.overwrite_representation,
    )
    head = fit_head(rep, method, seed=config.seed, num_reads=config.num_reads, num_sweeps=config.num_sweeps)
    Xtr = validate_matrix(rep.X_train, "X_train")
    Xte = validate_matrix(rep.X_test, "X_test", n_features=Xtr.shape[1])
    validate_predictor(head.predict_proba, Xtr.shape[1])

    # Evaluate the fitted classifier on the complete held-out test split first.
    # The explanation subset below is never used as a substitute for test metrics.
    all_probs = head.predict_proba(Xte)
    decision_threshold = head.fit_metadata.get("decision_threshold_probability")
    if all_probs.shape[1] == 2 and decision_threshold is not None:
        all_pred_class = (all_probs[:, 1] >= float(decision_threshold)).astype(int)
    else:
        all_pred_class = all_probs.argmax(axis=1)
    accuracy = float(accuracy_score(rep.y_test, all_pred_class))
    balanced_accuracy = float(balanced_accuracy_score(rep.y_test, all_pred_class))
    macro_f1 = float(f1_score(rep.y_test, all_pred_class, average="macro", zero_division=0))

    sample_idx = _select_indices(rep.y_test, min(config.explain_samples, len(rep.y_test)), config.seed)
    Xexp = Xte[sample_idx]
    y_true = rep.y_test[sample_idx]
    probs = all_probs[sample_idx]
    pred_class = all_pred_class[sample_idx]

    bg_a = _sample_background(Xtr, config.background_size, config.seed + 17)
    stability_backgrounds = [
        _sample_background(Xtr, config.background_size, config.seed + 104729 * (r + 1))
        for r in range(config.stability_repeats)
    ]
    d = Xtr.shape[1]
    shap_nsamples = int(config.shap_nsamples or (2 * d + 2048))
    rows, feature_rows = [], []
    captured_warnings = []

    for i, (row, true_y, cls) in enumerate(zip(Xexp, y_true, pred_class)):
        cls = int(cls)
        scalar = _decision_margin_predictor(head, cls, row)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shap_a = explain_kernel(scalar, bg_a, row[None, :], nsamples=shap_nsamples)
            shap_repeats = [
                explain_kernel(scalar, bg, row[None, :], nsamples=shap_nsamples)
                for bg in stability_backgrounds
            ]
            for warning in caught:
                captured_warnings.append({"sample": int(i), "stage": "shap", "category": warning.category.__name__, "message": str(warning.message)})
        shap_values = shap_a["values"][0]
        validate_explanation(shap_values, d, "SHAP")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lime_a = explain_one(scalar, Xtr, row, num_samples=config.lime_num_samples,
                                seed=config.seed + i * 37 + 1, feature_names=rep.feature_names)
            lime_repeats = [
                explain_one(scalar, Xtr, row, num_samples=config.lime_num_samples,
                            seed=config.seed + i * 37 + 2 + r, feature_names=rep.feature_names)
                for r in range(config.stability_repeats)
            ]
            for warning in caught:
                captured_warnings.append({"sample": int(i), "stage": "lime", "category": warning.category.__name__, "message": str(warning.message)})
        lime_values = np.asarray(lime_a["values"], dtype=np.float64)
        lime_weights = np.asarray(lime_a.get("weights", lime_a["values"]), dtype=np.float64)
        validate_explanation(lime_values, d, "LIME")

        shap_pred = float(shap_a["predictions"][0])
        shap_error = float(shap_a["additivity_error"][0])
        shap_faith = faithfulness_metrics(scalar, row, shap_values, bg_a,
                                          top_k=config.top_k, repeats=config.random_faithfulness_repeats,
                                          seed=config.seed + i * 101)
        lime_faith = faithfulness_metrics(scalar, row, lime_values, bg_a,
                                          top_k=config.top_k, repeats=config.random_faithfulness_repeats,
                                          seed=config.seed + i * 101 + 1)
        # Stability and cross-method agreement compare feature-importance
        # magnitudes, not signed coefficient order (which can make a negative
        # coefficient appear spuriously more important than a positive one).
        shap_stability = _mean_finite([
            rank_correlation(np.abs(shap_values), np.abs(rep_values["values"][0]))
            for rep_values in shap_repeats
        ])
        lime_stability = _mean_finite([
            rank_correlation(np.abs(lime_values), np.abs(rep_values["values"]))
            for rep_values in lime_repeats
        ])
        shap_jaccard = float(np.mean([
            topk_jaccard(shap_values, rep_values["values"][0], min(config.top_k, d))
            for rep_values in shap_repeats
        ]))
        lime_jaccard = float(np.mean([
            topk_jaccard(lime_values, rep_values["values"], min(config.top_k, d))
            for rep_values in lime_repeats
        ]))
        agreement = rank_correlation(np.abs(shap_values), np.abs(lime_values))
        rows.append({
            "dataset": dataset, "method": method, "seed": config.seed,
            "test_row_index": int(sample_idx[i]), "true_label": int(true_y),
            "predicted_class": cls, "predicted_class_probability": float(probs[i, cls]),
            "shap_base_value": float(shap_a["base_value"]),
            "explanation_target": "predicted_class_vs_strongest_competitor_score_margin",
            "shap_model_output": shap_pred, "shap_additivity_abs_error": shap_error,
            "shap_local_accuracy_status": local_accuracy_status(shap_error),
            "lime_local_r2": float(lime_a["local_r2"]),
            "lime_instance_prediction": float(lime_a["instance_prediction"]),
            "lime_local_prediction": float(lime_a["local_prediction"]),
            "lime_backend": lime_a["backend"], "shap_backend": shap_a["backend"],
            "shap_stability_spearman": shap_stability, "shap_stability_topk_jaccard": shap_jaccard,
            "lime_stability_spearman": lime_stability, "lime_stability_topk_jaccard": lime_jaccard,
            "shap_lime_rank_spearman": agreement,
            **{f"shap_{k}": v for k, v in shap_faith.items()},
            **{f"lime_{k}": v for k, v in lime_faith.items()},
        })
        for j, fname in enumerate(rep.feature_names):
            feature_rows.append({
                "dataset": dataset, "method": method, "seed": config.seed,
                "test_row_index": int(sample_idx[i]), "true_label": int(true_y),
                "predicted_class": cls, "predicted_class_probability": float(probs[i, cls]),
                "feature_index": j, "feature_name": fname, "feature_value": float(row[j]),
                "shap_value": float(shap_values[j]),
                "lime_contribution": float(lime_values[j]),
                "lime_weight_per_sd": float(lime_weights[j]),
            })

    out_dir = Path(config.output_dir) / dataset / method / f"seed_{config.seed}_{started.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    samples_df = pd.DataFrame(rows)
    attrs_df = pd.DataFrame(feature_rows)
    samples_df.to_csv(out_dir / "sample_metrics.csv", index=False)
    attrs_df.to_csv(out_dir / "feature_attributions.csv", index=False)

    numeric_cols = samples_df.select_dtypes(include=[np.number]).columns
    summary = {col: float(samples_df[col].mean()) for col in numeric_cols if col not in {"seed", "test_row_index", "true_label", "predicted_class"}}
    summary.update({"dataset_accuracy_on_test": accuracy, "dataset_balanced_accuracy_on_test": balanced_accuracy,
                    "dataset_macro_f1_on_test": macro_f1, "n_explained": int(len(samples_df)),
                    "n_features": int(d), "n_background": int(len(bg_a)), "shap_nsamples": shap_nsamples,
                    "lime_num_samples": config.lime_num_samples})
    pd.DataFrame([summary]).to_csv(out_dir / "summary_metrics.csv", index=False)
    plots = _plot_global(attrs_df, out_dir, rep.feature_names)

    manifest = {
        "dataset": dataset, "method": method, "seed": config.seed,
        "created_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "explanation_scope": (
            "classifier head over frozen ResNet18 512-dimensional penultimate features"
            if dataset == "resnet18_cifar10"
            else "classifier head over frozen BNN-DFA hidden representations"
    ),
        "explanation_target": "predicted_class_vs_strongest_competitor_score_margin; continuous model decision score rather than saturated probability",
        "not_explained": "raw pixels or raw input columns; attributions are for latent hidden features",
        "representation_path": rep.representation_path,
        "representation_sha256": rep.representation_sha256,
        "representation_source": rep.representation_source,
        "representation_metadata": _safe_json(rep.metadata),
        "head_fit_metadata": _safe_json(head.fit_metadata),
        "feature_names": rep.feature_names,
        "config": _safe_json(config.__dict__),
        "test_evaluation": {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "macro_f1": macro_f1,
                            "n_test": int(len(rep.y_test)),
                            "decision_threshold_probability": decision_threshold},
        "explanation_sample_indices": sample_idx.tolist(),
        "explanation_sampling": "deterministic class-stratified sample of held-out test indices; labels are used only to define/report the sample and are not used to fit or tune the explained head",
        "backend": {"shap": "shap.KernelExplainer", "lime": sorted(samples_df["lime_backend"].unique().tolist())},
        "captured_warnings": captured_warnings,
        "validation_summary": {
            "shap_additivity_pass_rate": float((samples_df["shap_local_accuracy_status"] == "pass").mean()),
            "mean_lime_local_r2": float(samples_df["lime_local_r2"].mean()),
            "mean_shap_lime_rank_spearman": float(samples_df["shap_lime_rank_spearman"].mean()),
            "mean_shap_stability_spearman": float(samples_df["shap_stability_spearman"].mean()),
            "mean_lime_stability_spearman": float(samples_df["lime_stability_spearman"].mean()),
        },
        "plots": plots,
        "interpretation_warning": "Ablation metrics measure sensitivity to the selected training-median replacement and are not causal effects. Latent features are not directly equivalent to original input features.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(_safe_json(manifest), indent=2), encoding="utf-8")
    _write_report(out_dir, manifest, samples_df)
    print(f"XAI run saved: {out_dir.resolve()}")
    print(f"Test accuracy={accuracy:.4f}; balanced_accuracy={balanced_accuracy:.4f}; macro_f1={macro_f1:.4f}")
    print(f"SHAP additivity pass rate={manifest['validation_summary']['shap_additivity_pass_rate']:.3f}; "
          f"LIME local R2={manifest['validation_summary']['mean_lime_local_r2']:.3f}")
    return out_dir


def _write_report(out_dir: Path, manifest: dict, samples_df: pd.DataFrame):
    v = manifest["validation_summary"]
    lines = [
        f"# XAI experiment report: {manifest['dataset']} / {manifest['method']}", "",
        f"- Seed: {manifest['seed']}", f"- Explanation scope: {manifest['explanation_scope']}",
        f"- Representation SHA-256: `{manifest['representation_sha256']}`",
        f"- Test accuracy: {manifest['test_evaluation']['accuracy']:.4f}",
        f"- Test balanced accuracy: {manifest['test_evaluation']['balanced_accuracy']:.4f}",
        f"- Test macro F1: {manifest['test_evaluation']['macro_f1']:.4f}", "",
        "## Explanation validation", "",
        f"- SHAP local-accuracy pass rate (absolute error <= 0.03): {v['shap_additivity_pass_rate']:.3f}",
        f"- Mean LIME weighted local R²: {v['mean_lime_local_r2']:.4f}",
        f"- Mean SHAP/LIME absolute-importance rank correlation: {v['mean_shap_lime_rank_spearman']:.4f}",
        f"- Mean SHAP stability rank correlation (independent background samples): {v['mean_shap_stability_spearman']:.4f}",
        f"- Mean LIME stability rank correlation (independent perturbation seeds): {v['mean_lime_stability_spearman']:.4f}", "",
        "## Files", "", "- `feature_attributions.csv`: per-sample, per-feature SHAP and LIME values.",
        "- `sample_metrics.csv`: explanation diagnostics and perturbation faithfulness metrics.",
        "- `summary_metrics.csv`: numeric summary.", "- `run_manifest.json`: configuration and provenance.", "",
        "## Important interpretation limits", "",
        manifest['interpretation_warning'],
        "These are pilot explanations on held-out test rows. They do not establish causal effects or population-level scientific conclusions. Multi-seed runs and prespecified analysis are required for study claims.",
    ]
    (out_dir / "VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
