"""Convolutional Direct Feedback Alignment (DFA) for CIFAR-10 BNN.

Mathematical Formulation:
-------------------------
In standard backpropagation, errors propagate sequentially:
    delta_l = (W_{l+1}^T delta_{l+1}) * sigma'(z_l)
creating a strict backward locking problem.

In Direct Feedback Alignment (Nøkland, 2016; Crafton et al., 2019):
Every hidden layer receives a direct, fixed random projection of the top-level
output error vector e = grad_{y_hat} L:
    delta_l = (e @ B_l) * sigma'(z_l)
where B_l is a fixed, non-trainable random matrix initialized at the start
of the experiment.

Feedback Dimensions for the CIFAR-10 BNN Architecture:
------------------------------------------------------
Classification error vector:
    e in R^{B x 10}

1. Conv1 Layer (3 -> 32, stride=2, padding=1, 3x3):
   Activation shape excluding batch: [32, 16, 16] -> 8,192 elements.
   Feedback matrix: B_conv1 in R^{10 x 8192}.
   Local feedback:
       delta_conv1_flat = e @ B_conv1                    # [B, 8192]
       delta_conv1 = reshape(delta_conv1_flat, [B, 32, 16, 16]) * (z_conv1 > 0)
   Local parameter gradients (adjoint of 2D convolution):
       grad_W_conv1 = conv2d_weight(x_input, W_conv1.shape, delta_conv1, stride=2, padding=1)
       grad_b_conv1 = sum_{b, h, w} delta_conv1          # [32]

2. Conv2 Layer (32 -> 64, stride=2, padding=1, 3x3):
   Activation shape excluding batch: [64, 8, 8] -> 4,096 elements.
   Feedback matrix: B_conv2 in R^{10 x 4096}.
   Local feedback:
       delta_conv2_flat = e @ B_conv2                    # [B, 4096]
       delta_conv2 = reshape(delta_conv2_flat, [B, 64, 8, 8]) * (z_conv2 > 0)
   Local parameter gradients:
       grad_W_conv2 = conv2d_weight(a_conv1, W_conv2.shape, delta_conv2, stride=2, padding=1)
       grad_b_conv2 = sum_{b, h, w} delta_conv2          # [64]

3. Dense Layer (64 -> 128):
   Activation shape excluding batch: [128] elements.
   Feedback matrix: B_dense in R^{10 x 128}.
   Local feedback:
       delta_dense = (e @ B_dense) * (z_dense > 0)       # [B, 128]
   Local parameter gradients:
       grad_W_fc1 = delta_dense^T @ a_gap                # [128, 64]
       grad_b_fc1 = sum_b delta_dense                    # [128]

4. Output Layer (128 -> 10):
   Standard loss gradient:
       grad_W_fc2 = e^T @ a_dense                        # [10, 128]
       grad_b_fc2 = sum_b e                              # [10]

Crucially:
- Local hidden layer learning signals are independent of error backpropagation
  through preceding or subsequent layers.
- Conv2 gradients do NOT backpropagate into Conv1.
- Dense gradients do NOT backpropagate into Conv2.
- Latent weights receive finite local updates, driving the BNN optimization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.grad as G


class CIFARDFA:
    """Manages fixed random feedback alignment matrices for CIFAR-10 BNN."""

    def __init__(
        self,
        num_classes: int = 10,
        conv1_shape: tuple[int, int, int] = (32, 16, 16),
        conv2_shape: tuple[int, int, int] = (64, 8, 8),
        dense_size: int = 128,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ):
        self.num_classes = num_classes
        self.conv1_shape = conv1_shape
        self.conv2_shape = conv2_shape
        self.dense_size = dense_size
        self.device = torch.device(device)

        self.conv1_flat_size = conv1_shape[0] * conv1_shape[1] * conv1_shape[2]  # 8192
        self.conv2_flat_size = conv2_shape[0] * conv2_shape[1] * conv2_shape[2]  # 4096

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))

        # Fixed random feedback matrices (never trained, detached)
        # Scaled by 1/sqrt(fan_in) to keep projection variance stable
        b_conv1 = torch.randn(
            (num_classes, self.conv1_flat_size),
            generator=generator,
            dtype=torch.float32,
        ) / (num_classes ** 0.5)

        b_conv2 = torch.randn(
            (num_classes, self.conv2_flat_size),
            generator=generator,
            dtype=torch.float32,
        ) / (num_classes ** 0.5)

        b_dense = torch.randn(
            (num_classes, dense_size),
            generator=generator,
            dtype=torch.float32,
        ) / (num_classes ** 0.5)

        self.feedback_conv1 = b_conv1.to(self.device).requires_grad_(False)
        self.feedback_conv2 = b_conv2.to(self.device).requires_grad_(False)
        self.feedback_dense = b_dense.to(self.device).requires_grad_(False)

    def compute_local_gradients(
        self,
        model: nn.Module,
        x_input: torch.Tensor,
        labels: torch.Tensor,
        forward_intermediates: tuple,
    ) -> dict[str, torch.Tensor]:
        """Compute layer-wise DFA parameter gradients without backpropagation.

        Parameters
        ----------
        model: CIFAR_BNN instance
        x_input: [B, 3, 32, 32] raw input image tensor
        labels: [B] integer class labels
        forward_intermediates: tuple returned by model.forward_with_intermediates(x_input)

        Returns
        -------
        Dictionary mapping parameter names to their manually calculated local gradients.
        """
        (
            outputs,
            conv1_pre,
            conv1_act,
            conv2_pre,
            conv2_act,
            gap_act,
            dense_pre,
            dense_act,
        ) = forward_intermediates

        batch_size = labels.size(0)

        # 1. Output error: exact gradient of CrossEntropyLoss w.r.t logits
        # dL/d(outputs) = (softmax(outputs) - one_hot(labels)) / batch_size
        output_error = CIFARDFAFunction.output_error(outputs, labels)

        # 2. Output layer (fc2) gradients (exact)
        grad_fc2_weight = output_error.T @ dense_act
        grad_fc2_bias = output_error.sum(dim=0)

        # 3. Dense layer (fc1) DFA feedback and local gradients
        dense_error_signal = output_error @ self.feedback_dense
        dense_delta = dense_error_signal * (dense_pre > 0).float()
        grad_fc1_weight = dense_delta.T @ gap_act
        grad_fc1_bias = dense_delta.sum(dim=0)

        # 4. Conv2 layer DFA feedback and local gradients
        conv2_error_signal_flat = output_error @ self.feedback_conv2
        conv2_delta = conv2_error_signal_flat.view(
            batch_size, *self.conv2_shape
        ) * (conv2_pre > 0).float()

        grad_conv2_weight = G.conv2d_weight(
            conv1_act,
            model.conv2.weight.shape,
            conv2_delta,
            stride=model.conv2.stride,
            padding=model.conv2.padding,
        )
        grad_conv2_bias = conv2_delta.sum(dim=(0, 2, 3))

        # 5. Conv1 layer DFA feedback and local gradients
        conv1_error_signal_flat = output_error @ self.feedback_conv1
        conv1_delta = conv1_error_signal_flat.view(
            batch_size, *self.conv1_shape
        ) * (conv1_pre > 0).float()

        grad_conv1_weight = G.conv2d_weight(
            x_input,
            model.conv1.weight.shape,
            conv1_delta,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
        )
        grad_conv1_bias = conv1_delta.sum(dim=(0, 2, 3))

        return {
            "conv1.weight": grad_conv1_weight,
            "conv1.bias": grad_conv1_bias,
            "conv2.weight": grad_conv2_weight,
            "conv2.bias": grad_conv2_bias,
            "fc1.weight": grad_fc1_weight,
            "fc1.bias": grad_fc1_bias,
            "fc2.weight": grad_fc2_weight,
            "fc2.bias": grad_fc2_bias,
        }

    def apply_gradients(
        self,
        model: nn.Module,
        gradients: dict[str, torch.Tensor],
    ):
        """Assign calculated local gradients directly to model parameter .grad attributes."""
        model.conv1.weight.grad = gradients["conv1.weight"]
        model.conv1.bias.grad = gradients["conv1.bias"]
        model.conv2.weight.grad = gradients["conv2.weight"]
        model.conv2.bias.grad = gradients["conv2.bias"]
        model.fc1.weight.grad = gradients["fc1.weight"]
        model.fc1.bias.grad = gradients["fc1.bias"]
        model.fc2.weight.grad = gradients["fc2.weight"]
        model.fc2.bias.grad = gradients["fc2.bias"]


class CIFARDFAFunction:
    """Utility functions for computing DFA error vectors."""

    @staticmethod
    def output_error(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the derivative of Cross-Entropy loss w.r.t unnormalized logits.

        For softmax probabilities p and 1-hot targets y:
            dL / d(logits) = (p - y) / N
        """
        probs = torch.softmax(outputs, dim=1)
        targets = torch.zeros_like(probs)
        targets.scatter_(1, labels.unsqueeze(1), 1.0)
        return (probs - targets) / labels.size(0)
