import math
from numbers import Integral, Real

import torch
import neal

from models.ls_heads import fit_ridge_classifier


def calculate_legacy_ls_scales(
    H,
    labels,
    num_classes=10
):
    """
    Calculate a natural scale for each binary classifier
    from the real-valued least-squares classifier.

    The scale matches the L2 norm of the real-valued
    classifier to the norm of a {-1,+1} binary vector.

        ||binary_weight|| = sqrt(hidden_size)

        alpha =
            ||LS_weight|| / sqrt(hidden_size)
    """

    H = H.float()

    hidden_size = H.shape[1]

    # ---------------------------------------------------------
    # Add bias feature for LS solution
    # ---------------------------------------------------------

    H_aug = torch.cat(
        [
            H,
            torch.ones(
                H.shape[0],
                1,
                dtype=H.dtype
            )
        ],
        dim=1
    )

    Y = torch.nn.functional.one_hot(
        labels,
        num_classes=num_classes
    ).float()

    # ---------------------------------------------------------
    # Real-valued least-squares classifier
    # ---------------------------------------------------------

    W_ls = torch.linalg.lstsq(
        H_aug,
        Y
    ).solution

    # W_ls shape:
    #
    # [hidden_size + 1, num_classes]

    W_ls_no_bias = W_ls[
        :hidden_size,
        :
    ]

    # ---------------------------------------------------------
    # Binary vector norm
    # ---------------------------------------------------------

    binary_norm = (
        hidden_size ** 0.5
    )

    # ---------------------------------------------------------
    # Class-specific scales
    # ---------------------------------------------------------

    scales = torch.zeros(
        num_classes,
        dtype=torch.float32
    )

    for class_id in range(num_classes):

        ls_norm = torch.linalg.vector_norm(
            W_ls_no_bias[:, class_id]
        )

        scale = (
            ls_norm
            / binary_norm
        )

        scales[class_id] = scale

    return scales




# Backwards-compatible legacy API for diagnostics and historical scripts.
def calculate_ls_scales(H, labels, num_classes=10):
    return calculate_legacy_ls_scales(H, labels, num_classes=num_classes)

def calculate_ridge_ls_scales(
    H,
    labels,
    num_classes=10,
    lambda_relative=1e-4,
):
    """Calculate QUBO alpha scales from the stable Ridge-LS fit.

    The target encoding remains {0,1}, matching the historical scale
    initialization; only the numerical LS solve is changed. The Ising objective
    and post-annealing scale/bias fitting are unchanged.
    """
    hidden_size = H.shape[1]
    W_ls = fit_ridge_classifier(
        H=H,
        labels=labels,
        num_classes=num_classes,
        lambda_relative=lambda_relative,
    )
    W_features = W_ls[:hidden_size, :]
    scales = (
        torch.linalg.vector_norm(W_features, dim=0)
        / (hidden_size ** 0.5)
    )
    return scales.to(dtype=torch.float32, device="cpu")


def build_ising_model(
    H,
    target,
    alpha
):
    """
    Build the Ising model for:

        min_w ||alpha * H w - target||^2

    subject to:

        w_i ∈ {-1,+1}

    The bias is eliminated analytically by centering H
    and target.

    alpha is fixed before QUBO optimization.
    """

    H = H.float()
    target = target.float()

    # ---------------------------------------------------------
    # Remove optimal continuous bias
    # ---------------------------------------------------------

    Hc = (
        H
        - H.mean(
            dim=0,
            keepdim=True
        )
    )

    yc = (
        target
        - target.mean()
    )

    # ---------------------------------------------------------
    # Objective:
    #
    # ||alpha Hc w - yc||²
    #
    # = alpha² wᵀHcᵀHc w
    #   - 2 alpha ycᵀHc w
    #   + constant
    # ---------------------------------------------------------

    gram = Hc.T @ Hc

    quadratic = (
        alpha ** 2
        * gram
    )

    linear = (
        -2.0
        * alpha
        * (Hc.T @ yc)
    )

    hidden_size = H.shape[1]

    h = {}
    J = {}

    # ---------------------------------------------------------
    # Linear Ising coefficients
    # ---------------------------------------------------------

    for i in range(hidden_size):

        h[i] = float(
            linear[i].item()
        )

    # ---------------------------------------------------------
    # Quadratic Ising coefficients
    # ---------------------------------------------------------

    for i in range(hidden_size):

        for j in range(i + 1, hidden_size):

            coefficient = (
                2.0
                * quadratic[i, j]
            ).item()

            if abs(coefficient) > 1e-12:

                J[(i, j)] = float(
                    coefficient
                )

    return h, J


def optimize_binary_class(
    H,
    target,
    alpha,
    num_reads=100,
    seed=None,
    num_sweeps=1000
):
    """
    Optimize one binary classifier using simulated annealing.
    """

    if not isinstance(num_reads, Integral) or isinstance(num_reads, bool) or num_reads <= 0:
        raise ValueError("num_reads must be a positive integer")
    if not isinstance(num_sweeps, Integral) or isinstance(num_sweeps, bool) or num_sweeps <= 0:
        raise ValueError("num_sweeps must be a positive integer")
    if (not isinstance(alpha, Real) or isinstance(alpha, bool)
            or not math.isfinite(float(alpha))):
        raise ValueError("alpha must be a finite real number")

    h, J = build_ising_model(
        H=H,
        target=target,
        alpha=alpha
    )

    sampler = (
        neal.SimulatedAnnealingSampler()
    )

    response = sampler.sample_ising(
        h,
        J,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed
    )

    try:
        sample_count = len(response)
    except Exception as exc:
        raise RuntimeError("simulated annealing returned an invalid response") from exc
    if sample_count == 0:
        raise RuntimeError("simulated annealing returned no samples")
    try:
        best_sample = response.first.sample
    except Exception as exc:
        raise RuntimeError("simulated annealing response has no first sample") from exc

    expected_variables = set(range(H.shape[1]))
    try:
        returned_variables = set(best_sample.keys())
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("simulated annealing first sample is not a variable mapping") from exc
    if not expected_variables.issubset(returned_variables):
        missing = sorted(expected_variables.difference(returned_variables))
        raise RuntimeError(f"simulated annealing sample is missing variables: {missing}")
    invalid = {variable: best_sample[variable] for variable in expected_variables
               if best_sample[variable] not in (-1, 1)}
    if invalid:
        raise RuntimeError(f"simulated annealing sample contains non-bipolar values: {invalid}")

    weights = torch.tensor(
        [float(best_sample[i]) for i in range(H.shape[1])],
        dtype=torch.float32
    )

    return weights


def optimize_binary_classifier(
    H,
    targets,
    scales,
    num_classes=10,
    num_reads=100,
    seed=None,
    num_sweeps=1000
):
    """
    Optimize one binary classifier per class.

    Each class uses its own scale alpha.

    Parameters
    ----------
    H:
        Hidden representations [N, hidden_size]

    targets:
        {-1,+1} targets [N, num_classes]

    scales:
        Class-specific QUBO scales [num_classes]
    """

    hidden_size = H.shape[1]

    binary_weights = torch.empty(
        hidden_size,
        num_classes,
        dtype=torch.float32
    )

    for class_id in range(num_classes):

        alpha = scales[class_id].item()

        print(
            f"Optimizing QUBO for class "
            f"{class_id} "
            f"(alpha={alpha:.6f})..."
        )

        target = (
            targets[:, class_id]
        )

        weights = optimize_binary_class(
            H=H,
            target=target,
            alpha=alpha,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed
        )

        binary_weights[
            :,
            class_id
        ] = weights

    return binary_weights
