import torch
import neal


def calculate_ls_scales(
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
    num_reads=100
):
    """
    Optimize one binary classifier using simulated annealing.
    """

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
        num_reads=num_reads
    )

    best_sample = (
        response.first.sample
    )

    weights = torch.tensor(
        [
            1.0
            if best_sample[i] == 1
            else -1.0
            for i in range(H.shape[1])
        ],
        dtype=torch.float32
    )

    return weights


def optimize_binary_classifier(
    H,
    targets,
    scales,
    num_classes=10,
    num_reads=100
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
            num_reads=num_reads
        )

        binary_weights[
            :,
            class_id
        ] = weights

    return binary_weights