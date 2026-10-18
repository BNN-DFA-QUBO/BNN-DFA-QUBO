"""Train-only standardization for binary classifier heads over BNN features.

Binary weights in {-1,+1} otherwise assign equal coefficients to hidden units
with potentially very different activation variances. Fit the transform on the
training representation only and reuse its parameters on validation/test data.
"""
from __future__ import annotations

import torch


def standardize_hidden_features(H_train, *others, eps: float = 1e-6):
    """Return standardized tensors and train-fitted scaling metadata.

    Constant/nearly-constant units are centered and assigned unit scale, so
    they become zeros instead of amplifying noise. Input tensors are detached
    and converted to float32 on the training tensor's device.
    """
    train = torch.as_tensor(H_train).detach()
    if train.ndim != 2 or train.shape[0] == 0 or train.shape[1] == 0:
        raise ValueError("H_train must be a non-empty [n_samples, n_features] matrix")
    device = train.device
    train = train.to(dtype=torch.float32)
    tensors = [train]
    for index, value in enumerate(others):
        tensor = torch.as_tensor(value).detach().to(device=device, dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[1] != train.shape[1]:
            raise ValueError(f"hidden feature matrix {index + 1} must have shape [n_samples, {train.shape[1]}]")
        tensors.append(tensor)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("hidden feature matrices must contain only finite values")
    if not torch.isfinite(torch.tensor(float(eps))) or eps <= 0:
        raise ValueError("eps must be finite and > 0")

    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False)
    constant_mask = std <= eps
    scale = torch.where(constant_mask, torch.ones_like(std), std)
    transformed = tuple((value - mean) / scale for value in tensors)
    metadata = {
        "mean": mean.detach().cpu(),
        "scale": scale.detach().cpu(),
        "constant_units": int(constant_mask.sum().item()),
        "transform": "train_fitted_hidden_mean_std",
        "eps": float(eps),
    }
    return (*transformed, metadata)
