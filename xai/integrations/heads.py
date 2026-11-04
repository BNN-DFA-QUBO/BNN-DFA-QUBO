"""Fit existing repository classifier heads on a validated frozen representation.

This XAI track targets the shared-representation classifier-head protocols. The
explained features are the BNN-DFA hidden representation (latent units), not raw
pixels or raw clinical/bank columns. Backbone training and classifier algorithms
are imported from existing project modules; no alternative model is substituted.
"""
from __future__ import annotations
import numpy as np
import torch

from xai.core.contracts import FittedHead, RepresentationBundle


def _torch(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _softmax(scores):
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim == 1:
        p1 = 1.0 / (1.0 + np.exp(-np.clip(arr, -40, 40)))
        return np.column_stack([1.0 - p1, p1])
    arr = arr - arr.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(arr, -80, 80))
    return exp / exp.sum(axis=1, keepdims=True)


def _make_head(bundle, method, score_fn, transform_fn=lambda x: x, feature_transform="identity", metadata=None):
    def predict_scores(x):
        tx = transform_fn(np.asarray(x, dtype=np.float64))
        out = score_fn(tx)
        return np.asarray(out, dtype=np.float64)
    def predict_proba(x):
        return _softmax(predict_scores(x))
    classes = sorted(np.unique(bundle.y_train).tolist())
    class_names = [str(int(c)) for c in classes]
    return FittedHead(bundle.dataset, method, predict_scores, predict_proba, class_names,
                      feature_transform, metadata or {})


def fit_head(bundle: RepresentationBundle, method: str, *, seed: int = 42,
             num_reads: int = 100, num_sweeps: int = 1000) -> FittedHead:
    method = method.lower()
    if method not in {"real_ls", "binarized_ls", "direct_binary", "ste_binary", "qubo"}:
        raise ValueError(f"Unsupported head method: {method}")
    Htr, ytr = _torch(bundle.X_train), torch.as_tensor(bundle.y_train, dtype=torch.long)
    Hva, yva = _torch(bundle.X_validation), torch.as_tensor(bundle.y_validation, dtype=torch.long)
    n_classes = int(max(np.max(bundle.y_train), np.max(bundle.y_validation))) + 1
    meta = {"n_train": int(len(ytr)), "n_validation": int(len(yva)), "n_features": int(Htr.shape[1]),
            "num_classes": n_classes, "seed": int(seed)}
    transform_fn = lambda x: np.asarray(x, dtype=np.float64)
    feature_transform = "identity"

    if method == "real_ls":
        if bundle.dataset == "stroke":
            # Match the repository's Stroke Real-LS implementation exactly:
            # unregularized NumPy least squares on raw hidden features with
            # an augmented intercept column.
            H = np.asarray(bundle.X_train, dtype=np.float64)
            design = np.column_stack((H, np.ones(len(H), dtype=np.float64)))
            weights, _, _, _ = np.linalg.lstsq(
                design, np.asarray(bundle.y_train, dtype=np.float64), rcond=None
            )
            score_fn = lambda x: (
                np.column_stack((np.asarray(x, dtype=np.float64),
                                np.ones(len(x), dtype=np.float64))) @ weights
            )
            meta["solver"] = "repository_numpy_lstsq_unregularized_augmented_intercept"
        else:
            from models.ls_heads import fit_ridge_classifier, predict_scores
            weights = fit_ridge_classifier(Htr, ytr, num_classes=n_classes, lambda_relative=1e-4)
            score_fn = lambda x: predict_scores(_torch(x), weights).numpy()
            meta["lambda_relative"] = 1e-4
    elif method == "binarized_ls":
        if bundle.dataset == "stroke":
            # Match Stroke's default Ridge -> sign -> training-only scale/bias
            # refit and train-only hidden-feature standardization.
            from utils.stroke_binary_features import standardize_hidden_features
            from experiments.stroke_experiments.stroke_dfa_binarized_ls import fit_binarized_ridge_ls
            Htr_s, Hva_s, Hte_s, tmeta = standardize_hidden_features(
                Htr, Hva, _torch(bundle.X_test)
            )
            binary_w, scale, bias, real_w = fit_binarized_ridge_ls(
                Htr_s, ytr, lambda_relative=1e-4
            )
            mean = tmeta["mean"].numpy()
            std = tmeta["scale"].numpy()
            transform_fn = lambda x: (np.asarray(x, dtype=np.float64) - mean) / std
            feature_transform = "train_fitted_hidden_mean_std"
            score_fn = lambda x: (
                (_torch(x) @ torch.as_tensor(binary_w, dtype=torch.float32))
                .reshape(-1).numpy() * float(scale) + float(bias)
            )
            meta.update({
                "solver": "repository_stroke_float64_augmented_svd_ridge",
                "lambda_relative": 1e-4,
                "scale": float(scale),
                "bias": float(bias),
                "standardized_constant_units": int(tmeta["constant_units"]),
                "scale_bias_fit": "training_only",
            })
        else:
            from models.ls_heads import fit_ridge_classifier, fit_binarized_ridge, binary_scores
            ridge = fit_ridge_classifier(Htr, ytr, num_classes=n_classes, lambda_relative=1e-4)
            w, scales, bias = fit_binarized_ridge(Htr, ytr, ridge, num_classes=n_classes)
            score_fn = lambda x: binary_scores(_torch(x), w, scales, bias).numpy()
            meta["base_solver"] = "stable_float64_svd_ridge"
            meta["scale_bias_fit"] = "training_only"
    elif method == "direct_binary":
        if bundle.dataset == "mnist":
            from experiments.MNIST_experiments.dfa_direct_binary_head import optimize_binary_head
            torch.manual_seed(int(seed))
            w, bias, train_acc = optimize_binary_head(Htr, ytr, num_classes=n_classes, steps=200, lr=0.05)
            score_fn = lambda x: (_torch(x) @ w + bias).numpy()
            meta.update({"steps": 200, "train_accuracy_percent": float(train_acc)})
        elif bundle.dataset == "bank":
            from experiments.bank_experiments.dfa_direct_binary_head import fit_direct_binary
            w, scale, bias = fit_direct_binary(Htr, ytr, num_classes=n_classes)
            score_fn = lambda x: (_torch(x) @ (w * scale.unsqueeze(0)) + bias).numpy()
            meta["fit"] = "class_centroid_difference_training_only"
        elif bundle.dataset == "cifar10":
            from experiments.CIFAR10_experiments.cifar_dfa_direct_binary import optimize_direct_binary_head
            w, bias, fit_time = optimize_direct_binary_head(Htr, ytr, num_classes=n_classes, steps=200, lr=0.05, seed=seed)
            score_fn = lambda x: (_torch(x) @ w + bias).numpy()
            meta.update({"steps": 200, "fit_time_sec": float(fit_time)})
        else:  # Stroke: exact class-mean sign rule with training-only standardization.
            from utils.stroke_binary_features import standardize_hidden_features
            Htr_s, Hva_s, _, tmeta = standardize_hidden_features(Htr, Hva, _torch(bundle.X_test))
            pos, neg = Htr_s[ytr == 1].mean(0), Htr_s[ytr == 0].mean(0)
            w = torch.sign(pos - neg); w[w == 0] = 1
            mean, scale = tmeta["mean"].numpy(), tmeta["scale"].numpy()
            transform_fn = lambda x: (np.asarray(x, dtype=np.float64) - mean) / scale
            feature_transform = "train_fitted_hidden_mean_std"
            score_fn = lambda x: (_torch(x) @ w).reshape(-1).numpy()
            meta["fit"] = "class_mean_sign_on_train_standardized_features"
    elif method == "ste_binary":
        if bundle.dataset == "mnist":
            # Reproduce the shared-validation STE checkpointing protocol locally
            # without importing that runner (which also imports optional QUBO code).
            import torch.nn.functional as F
            from experiments.MNIST_experiments.dfa_binary_head_ste import binary_ste
            torch.manual_seed(int(seed))
            hidden_size = Htr.shape[1]
            latent = (torch.randn(hidden_size, n_classes) * 0.01).requires_grad_()
            bias = torch.zeros(n_classes, requires_grad=True)
            optimizer = torch.optim.Adam([latent, bias], lr=0.01)
            best = None
            for epoch in range(50):
                optimizer.zero_grad()
                logits = Htr @ binary_ste(latent) + bias
                loss = F.cross_entropy(logits, ytr)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    val_logits = Hva @ binary_ste(latent) + bias
                    val_accuracy = float((val_logits.argmax(1) == yva).float().mean().item())
                    if best is None or val_accuracy > best["val_accuracy"]:
                        best = {"weights": binary_ste(latent).detach().clone(),
                                "bias": bias.detach().clone(),
                                "epoch": epoch + 1, "val_accuracy": val_accuracy}
            w, bias = best["weights"], best["bias"]
            score_fn = lambda x: (_torch(x) @ w + bias).numpy()
            meta.update({"epochs": 50, "best_epoch": int(best["epoch"]), "validation_accuracy": float(best["val_accuracy"])})
        elif bundle.dataset == "bank":
            from experiments.bank_experiments.dfa_binary_head_ste import train_ste_head
            w, scale, bias, threshold = train_ste_head(Htr, ytr, Hva, yva, num_classes=n_classes, steps=300, seed=seed)
            score_fn = lambda x: (_torch(x) @ (w * scale.unsqueeze(0)) + bias).numpy()
            meta.update({"steps": 300, "validation_threshold": float(threshold)})
        elif bundle.dataset == "cifar10":
            from experiments.CIFAR10_experiments.cifar_dfa_ste_binary import train_ste_with_validation
            best, fit_time = train_ste_with_validation(Htr, ytr, Hva, yva, epochs=50, lr=0.01, seed=seed)
            w, bias = best["weights"], best["bias"]
            score_fn = lambda x: (_torch(x) @ w + bias).numpy()
            meta.update({"epochs": 50, "best_epoch": int(best["epoch"]), "validation_accuracy": float(best["val_accuracy"]), "fit_time_sec": float(fit_time)})
        else:
            from utils.stroke_binary_features import standardize_hidden_features
            from experiments.stroke_experiments.stroke_dfa_ste_binary import _fit_candidate
            Htr_s, Hva_s, _, tmeta = standardize_hidden_features(Htr, Hva, _torch(bundle.X_test))
            candidates = [_fit_candidate(Htr_s, ytr.float(), Hva_s, yva.float(), seed + i * 1009, init)
                          for i, init in enumerate(("ridge", "class_mean", "random"))]
            best = max(candidates, key=lambda item: item["selection_key"])
            mean, scale = tmeta["mean"].numpy(), tmeta["scale"].numpy()
            transform_fn = lambda x: (np.asarray(x, dtype=np.float64) - mean) / scale
            feature_transform = "train_fitted_hidden_mean_std"
            w, bias = best["weights"], best["bias"]
            score_fn = lambda x: ((_torch(x) @ w) / (w.numel() ** 0.5) + float(bias)).reshape(-1).numpy()
            meta.update({"best_epoch": int(best["best_epoch"]), "initialization": best["initialization"],
                         "validation_balanced_accuracy": float(best["validation_balanced_accuracy"])})
    else:  # QUBO
        if bundle.dataset in {"mnist", "cifar10"}:
            from models.qubo_head import calculate_ridge_ls_scales, optimize_binary_classifier
            from experiments.MNIST_experiments.bnn_dfa_qubo import create_binary_targets, calculate_final_scale_and_bias
            targets = create_binary_targets(ytr, num_classes=n_classes)
            scales = calculate_ridge_ls_scales(Htr, ytr, num_classes=n_classes, lambda_relative=1e-4)
            scales = torch.where(scales > 1e-8, scales, torch.full_like(scales, 1e-8))
            w = optimize_binary_classifier(Htr, targets, scales, num_classes=n_classes,
                                           num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
            w, scales, bias = calculate_final_scale_and_bias(Htr, ytr, w, num_classes=n_classes)
            score_fn = lambda x: (_torch(x) @ (w * scales.unsqueeze(0)) + bias).numpy()
        elif bundle.dataset == "bank":
            from experiments.bank_experiments.tune_qubo_validation import tune_from_representations
            winner, w, scales, bias, rows, elapsed = tune_from_representations(
                Htr, ytr, Hva, yva, seed,
            )
            score_fn = lambda x: (_torch(x) @ (w * scales.unsqueeze(0)) + bias).numpy()
            meta.update({"validation_selected": True, "selected_num_reads": int(winner["num_reads"]),
                         "selected_num_sweeps": int(winner["num_sweeps"]), "grid_size": len(rows), "fit_time_sec": float(elapsed)})
        else:
            from utils.stroke_binary_features import standardize_hidden_features
            from experiments.stroke_experiments.stroke_dfa_qubo import train_qubo
            if not bundle.representation_path:
                raise ValueError("Stroke QUBO requires the validated on-disk representation artifact")
            result = train_qubo(bundle.representation_path, seed)
            Htr_s, Hva_s, _, tmeta = standardize_hidden_features(Htr, Hva, _torch(bundle.X_test))
            mean, scale = tmeta["mean"].numpy(), tmeta["scale"].numpy()
            transform_fn = lambda x: (np.asarray(x, dtype=np.float64) - mean) / scale
            feature_transform = "train_fitted_hidden_mean_std"
            w = torch.as_tensor(result["binary_weights"], dtype=torch.float32)
            alpha, bias = float(result["alpha"]), float(result["bias"])
            score_fn = lambda x: (alpha * (_torch(x) @ w) + bias).reshape(-1).numpy()
            meta.update({"selected_alpha": alpha, "validation_balanced_accuracy": result.get("validation_balanced_accuracy"),
                         "qubo_num_reads": result.get("qubo_num_reads"), "qubo_num_sweeps": result.get("qubo_num_sweeps")})
        meta.update({"num_reads": int(num_reads), "num_sweeps": int(num_sweeps), "solver": "dwave-neal simulated annealing"})

    head = _make_head(bundle, method, score_fn, transform_fn, feature_transform, meta)
    # Preserve the parent project's binary operating-point protocol: any
    # threshold is selected from validation data only, never from test labels.
    if bundle.dataset == "bank":
        from utils.bank.metrics import find_best_threshold
        val_prob = head.predict_proba(bundle.X_validation)[:, 1]
        threshold = float(find_best_threshold(bundle.y_validation, val_prob))
        head.fit_metadata["decision_threshold_probability"] = threshold
        head.fit_metadata["threshold_selection"] = "validation_balanced_accuracy"
    elif bundle.dataset == "stroke":
        from utils.stroke_metrics import find_best_threshold
        val_score = head.predict_scores(bundle.X_validation).reshape(-1)
        threshold_score, val_ba = find_best_threshold(bundle.y_validation, val_score)
        # The score-to-probability transform is sigmoid; convert the selected
        # raw-score threshold to the equivalent probability threshold.
        threshold_probability = float(1.0 / (1.0 + np.exp(-np.clip(threshold_score, -40, 40))))
        head.fit_metadata["decision_threshold_probability"] = threshold_probability
        head.fit_metadata["decision_threshold_score"] = float(threshold_score)
        head.fit_metadata["validation_balanced_accuracy"] = float(val_ba)
        head.fit_metadata["threshold_selection"] = "validation_balanced_accuracy"
    return head
