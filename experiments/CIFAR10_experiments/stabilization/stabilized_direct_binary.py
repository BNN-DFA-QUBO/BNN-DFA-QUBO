"""Direct Binary Head with diagnostic and stabilization variants.

Investigates:
1. Baseline: Unscaled tanh relaxation -> hard sign projection.
2. Stabilized: Scale & bias refit (consistent with Binarized LS and QUBO), preventing
   logit explosion and 2-class prediction collapse.
"""
from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from experiments.CIFAR10_experiments.cifar_dfa_direct_binary import (
    evaluate_direct_binary_head as evaluate_direct_binary_baseline,
    optimize_direct_binary_head as optimize_direct_binary_baseline,
    predict_direct_binary,
)


def optimize_direct_binary_stabilized(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int = 10,
    steps: int = 200,
    lr: float = 0.05,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Train direct binary head with tanh relaxation and refit scale & bias.

    This matches the consistent treatment in Binarized Least Squares and QUBO:
    finding optimal binary weights W in {-1, +1}^(d x C), then computing the
    analytical 1D scaling alpha_c and intercept beta_c on the training split.
    """
    if seed is not None:
        torch.manual_seed(seed)

    H = H_train.float()
    hidden_size = H.shape[1]

    latent = (torch.randn(hidden_size, num_classes, dtype=torch.float32) * 0.01).requires_grad_()
    bias = torch.zeros(num_classes, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([latent, bias], lr=lr)

    t0 = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad()
        w_relaxed = torch.tanh(latent)
        logits = H @ w_relaxed + bias
        loss = F.cross_entropy(logits, y_train.long())
        loss.backward()
        optimizer.step()
    fit_time = time.perf_counter() - t0

    with torch.no_grad():
        w_binary = torch.where(latent >= 0, torch.ones_like(latent), -torch.ones_like(latent))

        # Fit class-specific scales and biases analytically (consistent with Binarized LS and QUBO)
        targets = F.one_hot(y_train.long(), num_classes=num_classes).float()
        scales, biases = [], []
        for class_id in range(num_classes):
            z = H @ w_binary[:, class_id]
            target = targets[:, class_id]
            z_mean, target_mean = z.mean(), target.mean()
            z_centered, target_centered = z - z_mean, target - target_mean
            denominator = z_centered @ z_centered
            if denominator.item() > 1e-12:
                scale_raw = (z_centered @ target_centered) / denominator
            else:
                scale_raw = torch.tensor(1.0, dtype=z.dtype)

            if scale_raw.item() < 0:
                w_binary[:, class_id] *= -1
                scale = -scale_raw
            else:
                scale = scale_raw
            scales.append(scale)
            biases.append(target_mean - scale_raw * z_mean)

        final_scales = torch.stack(scales)
        final_biases = torch.stack(biases)

    return w_binary, final_scales, final_biases, fit_time


def evaluate_direct_binary_stabilized(
    H_train: torch.Tensor,
    y_train: torch.Tensor,
    H_val: torch.Tensor,
    y_val: torch.Tensor,
    H_test: torch.Tensor,
    y_test: torch.Tensor,
    steps: int = 200,
    lr: float = 0.05,
    seed: int | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, float], float]:
    w_bin, scales, biases, fit_time = optimize_direct_binary_stabilized(
        H_train, y_train, num_classes=10, steps=steps, lr=lr, seed=seed
    )

    def acc(H, y):
        scores = (H.float() @ w_bin) * scales.unsqueeze(0) + biases.unsqueeze(0)
        return float((scores.argmax(dim=1) == y).float().mean().item() * 100.0)

    accuracies = {
        "train": acc(H_train, y_train),
        "validation": acc(H_val, y_val),
        "test": acc(H_test, y_test),
    }
    return (w_bin, scales, biases), accuracies, fit_time
