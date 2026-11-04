import numpy as np

from xai.explainers.shap_explainer import explain_kernel
from xai.explainers.lime_explainer import explain_one
from xai.evaluation.metrics import faithfulness_metrics, rank_correlation, topk_jaccard
from xai.evaluation.validation import validate_explanation
from xai.core.contracts import validate_predictor


def _predict_proba(x):
    x = np.asarray(x, dtype=float)
    logit = 1.25 * x[:, 0] - 0.8 * x[:, 1] + 0.2 * x[:, 2]
    p1 = 1.0 / (1.0 + np.exp(-logit))
    return np.column_stack([1.0 - p1, p1])


def test_predictor_probability_contract():
    validate_predictor(_predict_proba, 3)


def test_kernel_shap_additivity_on_linear_logit_probability():
    rng = np.random.default_rng(4)
    background = rng.normal(size=(30, 3))
    x = rng.normal(size=(1, 3))
    result = explain_kernel(lambda z: _predict_proba(z)[:, 1], background, x, nsamples=64)
    validate_explanation(result["values"][0], 3, "SHAP")
    assert result["additivity_error"][0] < 1e-4


def test_lime_local_surrogate_and_finite_values():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(250, 3))
    x = train[0]
    result = explain_one(lambda z: _predict_proba(z)[:, 1], train, x,
                         num_samples=800, seed=12, feature_names=["a", "b", "c"])
    validate_explanation(result["values"], 3, "LIME")
    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0)
    train_std = np.where(train_std > 1e-8, train_std, 1.0)
    expected_contribution = result["weights"] * ((x - train_mean) / train_std)
    np.testing.assert_allclose(result["values"], expected_contribution, atol=1e-10)
    assert np.isfinite(result["local_r2"])
    assert result["backend"] in {"internal_lime_weighted_ridge", "lime.lime_tabular.LimeTabularExplainer"}


def test_faithfulness_metrics_and_rank_metrics():
    rng = np.random.default_rng(8)
    background = rng.normal(size=(100, 3))
    x = np.array([2.0, -1.0, 0.1])
    attr = np.array([2.0, -0.8, 0.1])
    result = faithfulness_metrics(lambda z: _predict_proba(z)[:, 1], x, attr, background,
                                  top_k=1, repeats=25, seed=1)
    assert result["topk_ablation_abs_change"] >= 0
    assert result["random_ablation_abs_change_mean"] >= 0
    assert topk_jaccard(attr, attr, 2) == 1.0
    assert rank_correlation(attr, attr) > 0.99


def test_stroke_ls_adapters_match_repository_specific_protocols():
    from xai.core.contracts import RepresentationBundle
    from xai.integrations.heads import fit_head
    from utils.stroke_binary_features import standardize_hidden_features
    from experiments.stroke_experiments.stroke_dfa_binarized_ls import fit_binarized_ridge_ls

    rng = np.random.default_rng(2026)
    X_train = rng.normal(size=(80, 9))
    y_train = rng.integers(0, 2, size=80)
    y_train[:2] = np.array([0, 1])
    X_val = rng.normal(size=(30, 9))
    y_val = rng.integers(0, 2, size=30)
    y_val[:2] = np.array([0, 1])
    X_test = rng.normal(size=(25, 9))
    y_test = rng.integers(0, 2, size=25)
    y_test[:2] = np.array([0, 1])
    bundle = RepresentationBundle(
        dataset="stroke", X_train=X_train, y_train=y_train,
        X_validation=X_val, y_validation=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=[f"latent_{i}" for i in range(9)],
        representation_path="synthetic-stroke-representation.pt",
        representation_sha256="synthetic", representation_source="unit_test", metadata={},
    )

    real_head = fit_head(bundle, "real_ls", seed=9)
    design = np.column_stack((X_train, np.ones(len(X_train))))
    expected_weights = np.linalg.lstsq(design, y_train.astype(float), rcond=None)[0]
    expected_scores = np.column_stack((X_test, np.ones(len(X_test)))) @ expected_weights
    np.testing.assert_allclose(real_head.predict_scores(X_test).reshape(-1), expected_scores, atol=1e-10)

    bin_head = fit_head(bundle, "binarized_ls", seed=9)
    import torch
    Htr_s, _ = standardize_hidden_features(torch.tensor(X_train, dtype=torch.float32))
    expected_w, expected_scale, expected_bias, _ = fit_binarized_ridge_ls(
        Htr_s, torch.tensor(y_train, dtype=torch.long), lambda_relative=1e-4
    )
    Htr_float = torch.tensor(X_train, dtype=torch.float32)
    mean = Htr_float.mean(dim=0).numpy()
    std = Htr_float.std(dim=0, unbiased=False).numpy()
    std = np.where(std <= 1e-6, 1.0, std)
    Xtest_s = (X_test - mean) / std
    expected_bin_scores = Xtest_s @ expected_w * expected_scale + expected_bias
    np.testing.assert_allclose(bin_head.predict_scores(X_test).reshape(-1), expected_bin_scores, atol=1e-5)
    assert bin_head.fit_metadata["threshold_selection"] == "validation_balanced_accuracy"


def test_cifar10_head_adapters_use_current_256_feature_contract():
    """Exercise CIFAR-10 head adapters against the repository's current width.

    The current CIFAR BNN-DFA representation is 256-dimensional (not the
    historical 128-dimensional value in a few older test comments). This is a
    synthetic adapter regression test, not a claim of real CIFAR data training.
    """
    import importlib
    from xai.core.contracts import RepresentationBundle
    from xai.integrations.heads import fit_head

    rng = np.random.default_rng(20261104)

    def make_split(n):
        x = rng.normal(0.0, 0.5, size=(n, 256)).astype(np.float64)
        y = np.tile(np.arange(10), (n + 9) // 10)[:n]
        rng.shuffle(y)
        x[np.arange(n), y] += 1.5
        return x, y.astype(np.int64)

    x_train, y_train = make_split(300)
    x_val, y_val = make_split(100)
    x_test, y_test = make_split(100)
    bundle = RepresentationBundle(
        dataset="cifar10",
        X_train=x_train, y_train=y_train,
        X_validation=x_val, y_validation=y_val,
        X_test=x_test, y_test=y_test,
        feature_names=[f"latent_{i:03d}" for i in range(256)],
        representation_path="synthetic-cifar10-test-only.pt",
        representation_sha256="synthetic-test-only",
        representation_source="unit_test_synthetic",
        metadata={"synthetic_test_only": True, "representation_dim": 256},
    )

    methods = ("real_ls", "binarized_ls", "direct_binary", "ste_binary")
    for method in methods:
        head = fit_head(bundle, method, seed=42)
        probabilities = head.predict_proba(x_test[:7])
        assert probabilities.shape == (7, 10)
        assert np.isfinite(probabilities).all()
        assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
        assert head.fit_metadata["n_features"] == 256
        assert head.fit_metadata["num_classes"] == 10

    # The QUBO entry point is deliberately imported independently; solving it
    # requires dimod/dwave-neal and is separately exercised in the real project
    # environment where those root requirements are installed.
    entry_points = (
        "xai_real_ls", "xai_binarized_ls", "xai_direct_binary",
        "xai_ste_binary", "xai_qubo",
    )
    for module in entry_points:
        importlib.import_module(f"xai.experiments.cifar10.{module}")


def test_decision_margin_explanation_target_avoids_probability_saturation():
    from types import SimpleNamespace
    from xai.core.runner import _decision_margin_predictor

    def scores(x):
        x = np.asarray(x, dtype=float)
        # Class 0 is the selected class at the reference row; class 1 is the
        # strongest competitor there. Probability softmax would be saturated.
        return np.column_stack((1000.0 + 3.0 * x[:, 0], -1000.0 + 2.0 * x[:, 1], np.full(len(x), -1200.0)))

    head = SimpleNamespace(predict_scores=scores)
    reference = np.array([0.0, 0.0])
    margin = _decision_margin_predictor(head, 0, reference)
    batch = np.array([[1.0, 2.0], [-1.0, 1.0]])
    expected = scores(batch)[:, 0] - scores(batch)[:, 1]
    np.testing.assert_allclose(margin(batch), expected)
    assert np.isfinite(margin(batch)).all()


def test_all_non_qubo_heads_smoke_across_all_four_dataset_tracks():
    """Exercise each supported non-QUBO head adapter on every dataset contract."""
    from xai.core.contracts import RepresentationBundle
    from xai.integrations.heads import fit_head

    rng = np.random.default_rng(20261105)
    cases = (("mnist", 128, 10), ("stroke", 64, 2),
             ("bank", 64, 2), ("cifar10", 256, 10))
    for dataset, n_features, n_classes in cases:
        def split(n):
            y = np.tile(np.arange(n_classes), (n + n_classes - 1) // n_classes)[:n]
            rng.shuffle(y)
            x = rng.normal(0.0, 0.5, size=(n, n_features))
            for cls in range(n_classes):
                x[y == cls, cls] += 1.2
            return x.astype(np.float64), y.astype(np.int64)

        x_train, y_train = split(180)
        x_val, y_val = split(60)
        x_test, y_test = split(40)
        bundle = RepresentationBundle(
            dataset=dataset, X_train=x_train, y_train=y_train,
            X_validation=x_val, y_validation=y_val, X_test=x_test, y_test=y_test,
            feature_names=[f"latent_{i:03d}" for i in range(n_features)],
            representation_path=f"synthetic-{dataset}-test-only.pt",
            representation_sha256="synthetic-test-only", representation_source="unit_test_synthetic",
            metadata={"synthetic_test_only": True},
        )
        for method in ("real_ls", "binarized_ls", "direct_binary", "ste_binary"):
            head = fit_head(bundle, method, seed=42)
            probabilities = head.predict_proba(x_test[:8])
            assert probabilities.shape == (8, n_classes), (dataset, method, probabilities.shape)
            assert np.isfinite(probabilities).all(), (dataset, method)
            assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
            if dataset in {"bank", "stroke"}:
                assert "decision_threshold_probability" in head.fit_metadata
