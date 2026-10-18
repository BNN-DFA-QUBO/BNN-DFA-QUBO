"""QUBO/Ising binary classifier head for Stroke DFA representations.

Hidden units are standardized from training representations only. A bounded
scale sweep combines simulated-annealing samples with class-mean, Ridge, and
random training-only binary starts refined under the training Ising energy.
Candidates are ranked using validation balanced accuracy, F1, PR-AUC, and
ROC-AUC, with validation-only threshold selection. Test labels do not take part
in calibration, candidate construction, ranking, or threshold selection.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import neal

from models.ls_heads import fit_ridge_regression
from utils.stroke_representation import load_representation
from utils.stroke_binary_features import standardize_hidden_features
from utils.stroke_metrics import find_best_threshold, calculate_metrics
from utils.seed import set_seed
from utils.stroke_config import (
    QUBO_NUM_READS,
    QUBO_NUM_SWEEPS,
    QUBO_RANDOM_SEED_STARTS,
    QUBO_ALPHA_MULTIPLIERS,
    QUBO_TOP_CANDIDATES_PER_SCALE,
)


def build_ising_model(H, target, alpha):
    """Build the Ising energy equivalent to centered signed-target squared error."""
    H64 = torch.as_tensor(H).detach().to(device="cpu", dtype=torch.float64)
    target64 = torch.as_tensor(target).detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    if H64.ndim != 2 or H64.shape[0] != target64.shape[0] or H64.shape[0] == 0:
        raise ValueError("QUBO requires aligned, non-empty features and targets")
    if not torch.isfinite(H64).all() or not torch.isfinite(target64).all():
        raise ValueError("QUBO requires finite features and targets")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("QUBO alpha must be finite and positive")

    H_centered = H64 - H64.mean(dim=0, keepdim=True)
    target_centered = target64 - target64.mean()
    gram = H_centered.T @ H_centered
    quadratic = (float(alpha) ** 2) * gram
    linear = -2.0 * float(alpha) * (H_centered.T @ target_centered)

    hidden_size = H64.shape[1]
    h = {i: float(linear[i].item()) for i in range(hidden_size)}
    J = {}
    for i in range(hidden_size):
        for j in range(i + 1, hidden_size):
            coefficient = 2.0 * float(quadratic[i, j].item())
            if abs(coefficient) > 1e-12:
                J[(i, j)] = coefficient
    return h, J


def _binary_sign(values, fallback=None):
    signs = torch.sign(torch.as_tensor(values, dtype=torch.float32))
    if fallback is not None:
        signs = torch.where(signs == 0, torch.sign(fallback), signs)
    return torch.where(signs == 0, torch.ones_like(signs), signs)


def _ising_energy(bits, h, J):
    return float(
        sum(float(h.get(i, 0.0)) * int(bits[i]) for i in range(len(bits)))
        + sum(float(value) * int(bits[i]) * int(bits[j]) for (i, j), value in J.items())
    )


def _refine_ising_bits(bits, h, J, max_passes=50):
    """Greedy single-spin descent using only the training-derived Ising energy."""
    spins = [int(value) for value in bits]
    neighbors = [[] for _ in spins]
    for (i, j), coupling in J.items():
        neighbors[i].append((j, float(coupling)))
        neighbors[j].append((i, float(coupling)))

    for _ in range(int(max_passes)):
        changed = False
        for i in range(len(spins)):
            local_field = float(h.get(i, 0.0)) + sum(
                coupling * spins[j] for j, coupling in neighbors[i]
            )
            delta_energy = -2.0 * spins[i] * local_field
            if delta_energy < -1e-10:
                spins[i] *= -1
                changed = True
        if not changed:
            break
    return tuple(spins)


def optimize_binary_class(H, target, alpha, num_reads, seed):
    """Return the lowest-energy binary {-1,+1} weight vector from one solve."""
    h, J = build_ising_model(H, target, alpha)
    response = neal.SimulatedAnnealingSampler().sample_ising(
        h, J, num_reads=int(num_reads), seed=int(seed)
    )
    sample = response.first.sample
    return torch.tensor(
        [1.0 if sample[i] == 1 else -1.0 for i in range(H.shape[1])],
        dtype=torch.float32,
    )


def sample_binary_candidates(
    H,
    target,
    base_alpha,
    num_reads,
    seed,
    alpha_multipliers=QUBO_ALPHA_MULTIPLIERS,
    top_k=QUBO_TOP_CANDIDATES_PER_SCALE,
    num_sweeps=QUBO_NUM_SWEEPS,
    random_seed_starts=QUBO_RANDOM_SEED_STARTS,
):
    """Collect distinct low-energy weight candidates across the alpha sweep."""
    H_cpu = torch.as_tensor(H).detach().to(device="cpu", dtype=torch.float32)
    target_cpu = torch.as_tensor(target).detach().reshape(-1).to(device="cpu", dtype=torch.float32)
    if top_k < 1 or num_reads < 1 or num_sweeps < 1:
        raise ValueError("top_k, num_reads, and num_sweeps must be positive")
    if random_seed_starts < 0:
        raise ValueError("random_seed_starts must be non-negative")
    if not alpha_multipliers:
        raise ValueError("At least one QUBO alpha multiplier is required")

    candidates = []
    seen = set()
    sampler = neal.SimulatedAnnealingSampler()
    hidden_size = H_cpu.shape[1]
    rng = np.random.default_rng(int(seed))

    positive_mask = target_cpu > 0
    negative_mask = target_cpu < 0
    if not positive_mask.any() or not negative_mask.any():
        raise ValueError("QUBO candidate search requires signed targets from both classes")
    class_mean_direction = (
        H_cpu[positive_mask].mean(dim=0) - H_cpu[negative_mask].mean(dim=0)
    )
    class_mean_seed = _binary_sign(class_mean_direction)
    ridge_coefficients = fit_ridge_regression(
        H_cpu, target_cpu, lambda_relative=1e-3
    )[:-1, 0].to(dtype=torch.float32)
    ridge_seed = _binary_sign(ridge_coefficients, class_mean_direction)

    for scale_index, multiplier in enumerate(alpha_multipliers):
        multiplier = float(multiplier)
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("QUBO alpha multipliers must be finite and positive")
        alpha = float(base_alpha) * multiplier
        h, J = build_ising_model(H_cpu, target_cpu, alpha)
        response = sampler.sample_ising(
            h,
            J,
            num_reads=int(num_reads),
            num_sweeps=int(num_sweeps),
            seed=int(seed) + 104729 * scale_index,
        )
        variable_positions = {variable: i for i, variable in enumerate(response.variables)}
        energies = np.asarray(response.record.energy, dtype=np.float64)
        samples = np.asarray(response.record.sample)
        for index in np.argsort(energies)[: min(int(top_k), len(energies))]:
            row = samples[index]
            bits = tuple(
                1 if int(row[variable_positions[i]]) == 1 else -1
                for i in range(hidden_size)
            )
            # Keep both the annealer sample and its training-energy coordinate
            # refinement. Deduplication preserves candidate diversity across scales.
            alternatives = (
                (bits, "annealing"),
                (_refine_ising_bits(bits, h, J), "coordinate_refinement"),
            )
            for candidate_bits, source in alternatives:
                if candidate_bits in seen:
                    continue
                seen.add(candidate_bits)
                candidates.append({
                    "weights": torch.tensor(candidate_bits, dtype=torch.float32),
                    "alpha": alpha,
                    "alpha_multiplier": multiplier,
                    "energy": _ising_energy(candidate_bits, h, J),
                    "candidate_source": source,
                })

        # Seed the discrete search with directions learned from training rows,
        # plus reproducible random binary vectors. Refine under this scale's
        # training-derived Ising energy; validation ranks candidates afterward.
        seeded_vectors = [
            ("class_mean_seed", tuple(int(v) for v in class_mean_seed.tolist())),
            ("ridge_seed", tuple(int(v) for v in ridge_seed.tolist())),
        ]
        for random_index in range(int(random_seed_starts)):
            random_bits = tuple(
                int(v) for v in rng.choice(np.array([-1, 1]), size=hidden_size)
            )
            seeded_vectors.append((f"random_seed_{random_index}", random_bits))

        for source, seed_bits in seeded_vectors:
            refined_bits = _refine_ising_bits(seed_bits, h, J, max_passes=50)
            for candidate_bits, candidate_source in (
                (seed_bits, source),
                (refined_bits, f"{source}_qubo_refined"),
            ):
                if candidate_bits in seen:
                    continue
                seen.add(candidate_bits)
                candidates.append({
                    "weights": torch.tensor(candidate_bits, dtype=torch.float32),
                    "alpha": alpha,
                    "alpha_multiplier": multiplier,
                    "energy": _ising_energy(candidate_bits, h, J),
                    "candidate_source": candidate_source,
                })
    if not candidates:
        raise RuntimeError("QUBO annealing returned no distinct binary candidates")
    return candidates


def calculate_alpha(H, labels, lambda_relative=1e-4):
    """Estimate a finite QUBO score scale by signed-target Ridge regression."""
    H64 = torch.as_tensor(H).detach().to(device="cpu", dtype=torch.float64)
    labels64 = torch.as_tensor(labels).detach().reshape(-1).to(
        device="cpu", dtype=torch.float64
    )
    if H64.ndim != 2 or H64.shape[0] != labels64.shape[0] or H64.shape[0] == 0:
        raise ValueError("QUBO scale calibration requires aligned non-empty hidden features and labels")
    if not torch.isfinite(H64).all() or not torch.isfinite(labels64).all():
        raise ValueError("QUBO scale calibration requires finite features and labels")
    if not torch.isin(labels64, torch.tensor([0.0, 1.0], dtype=torch.float64)).all():
        raise ValueError("QUBO scale calibration expects binary labels encoded as {0,1}")

    signed_targets = 2.0 * labels64 - 1.0
    try:
        coefficients = fit_ridge_regression(
            H64, signed_targets, lambda_relative=lambda_relative
        )[:-1, 0]
    except ValueError as exc:
        if "Feature energy is zero or non-finite" in str(exc):
            raise ValueError(
                "QUBO scale calibration is degenerate (hidden features have zero energy)."
            ) from exc
        raise
    alpha = torch.linalg.vector_norm(coefficients) / (H64.shape[1] ** 0.5)
    alpha_value = float(alpha.item())
    if not np.isfinite(alpha_value) or alpha_value <= 1e-8:
        raise ValueError(
            "QUBO scale calibration is degenerate (alpha is zero or near zero). "
            "Inspect the saved representation for collapsed or non-informative hidden features."
        )
    return alpha_value


def recover_bias(H, target, weights, alpha):
    return (
        target.mean()
        - alpha * (H.mean(dim=0) @ weights)
    ).item()


def train_qubo(representation_path, seed):
    set_seed(seed)
    data = load_representation(representation_path)
    H_train, H_val, H_test, transform = standardize_hidden_features(
        data["H_train"], data["H_val"], data["H_test"]
    )
    y_train = torch.as_tensor(data["y_train"]).float().reshape(-1)
    y_val = torch.as_tensor(data["y_val"]).float().reshape(-1)
    y_test = torch.as_tensor(data["y_test"]).float().reshape(-1)
    target_train = 2.0 * y_train - 1.0

    base_alpha = calculate_alpha(H_train, y_train)
    candidates = sample_binary_candidates(
        H=H_train,
        target=target_train,
        base_alpha=base_alpha,
        num_reads=QUBO_NUM_READS,
        seed=seed,
    )

    best = None
    for candidate in candidates:
        weights = candidate["weights"]
        alpha = candidate["alpha"]
        bias = recover_bias(H_train, target_train, weights, alpha)
        val_scores = alpha * (H_val @ weights) + bias
        threshold, val_ba = find_best_threshold(y_val.numpy(), val_scores.numpy())
        val_metrics = calculate_metrics(y_val.numpy(), val_scores.numpy(), threshold)
        # Compare the full validation metric profile instead of optimizing one
        # metric in isolation. Every component is on [0,1]; balanced accuracy is
        # the first tie-breaker. Test labels are never used for this ranking.
        # Energies from different alpha multipliers are on different scales.
        validation_selection_score = float(np.mean([
            val_ba,
            val_metrics["f1"],
            val_metrics["pr_auc"],
            val_metrics["roc_auc"],
        ]))
        key = (
            validation_selection_score,
            float(val_ba),
            float(val_metrics["f1"]),
            float(val_metrics["pr_auc"]),
            float(val_metrics["roc_auc"]),
        )
        if best is None or key > best["selection_key"]:
            best = {
                "selection_key": key,
                "validation_selection_score": validation_selection_score,
                "candidate": candidate,
                "weights": weights,
                "alpha": alpha,
                "bias": float(bias),
                "threshold": float(threshold),
                "validation_balanced_accuracy": float(val_ba),
                "validation_f1": float(val_metrics["f1"]),
                "validation_pr_auc": float(val_metrics["pr_auc"]),
                "validation_roc_auc": float(val_metrics["roc_auc"]),
            }

    test_scores = best["alpha"] * (H_test @ best["weights"]) + best["bias"]
    metrics = calculate_metrics(y_test.numpy(), test_scores.numpy(), best["threshold"])
    return {
        "method": "QUBO Binary",
        "seed": int(data.get("training_seed", data.get("seed", seed))),
        "head_seed": int(seed),
        "threshold": best["threshold"],
        "validation_balanced_accuracy": best["validation_balanced_accuracy"],
        "validation_selection_score": best["validation_selection_score"],
        "validation_f1": best["validation_f1"],
        "validation_pr_auc": best["validation_pr_auc"],
        "validation_roc_auc": best["validation_roc_auc"],
        "binary_weights": best["weights"].numpy(),
        "alpha": float(best["alpha"]),
        "base_alpha": float(base_alpha),
        "alpha_multiplier": float(best["candidate"]["alpha_multiplier"]),
        "bias": float(best["bias"]),
        "qubo_num_reads": QUBO_NUM_READS,
        "qubo_num_reads_per_scale": QUBO_NUM_READS,
        "qubo_num_sweeps": QUBO_NUM_SWEEPS,
        "qubo_random_seed_starts": QUBO_RANDOM_SEED_STARTS,
        "qubo_total_reads": QUBO_NUM_READS * len(QUBO_ALPHA_MULTIPLIERS),
        "qubo_alpha_multipliers": list(QUBO_ALPHA_MULTIPLIERS),
        "qubo_candidate_count": len(candidates),
        "qubo_energy": float(best["candidate"]["energy"]),
        "qubo_candidate_source": best["candidate"]["candidate_source"],
        "hidden_feature_transform": transform["transform"],
        "standardized_constant_units": transform["constant_units"],
        **metrics,
    }


if __name__ == "__main__":
    import argparse
    from utils.stroke_data import DATA_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", type=Path, default=DATA_DIR / "stroke_dfa_representation_seed_42.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_qubo(args.representation, args.seed)
    print("\nQUBO Binary")
    print(f"Validation BA: {result['validation_balanced_accuracy'] * 100:.4f}%")
    print(f"Test BA: {result['balanced_accuracy'] * 100:.4f}%")
    print(f"F1: {result['f1'] * 100:.4f}%")
    print(f"ROC-AUC: {result['roc_auc'] * 100:.4f}%")
    print(f"PR-AUC: {result['pr_auc'] * 100:.4f}%")
    print(f"Base alpha: {result['base_alpha']:.6f}")
    print(f"Selected alpha: {result['alpha']:.6f} (multiplier {result['alpha_multiplier']})")
    print(f"Candidates: {result['qubo_candidate_count']}")
