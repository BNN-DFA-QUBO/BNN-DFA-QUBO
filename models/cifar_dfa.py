"""Convolutional Direct Feedback Alignment (DFA) for CIFAR-10 BNN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.grad as G


class CIFARDFA:
    """Manages fixed random feedback alignment matrices for CIFAR-10."""

    def __init__(
        self,
        num_classes: int = 10,
        conv1_shape: tuple[int, int, int] = (32, 16, 16),
        conv2_shape: tuple[int, int, int] = (64, 8, 8),
        dense_size: int = 256,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ):
        self.num_classes = num_classes
        self.conv1_shape = conv1_shape
        self.conv2_shape = conv2_shape
        self.dense_size = dense_size
        self.device = torch.device(device)

        self.conv1_flat_size = (
            conv1_shape[0] * conv1_shape[1] * conv1_shape[2]
        )
        self.conv2_flat_size = (
            conv2_shape[0] * conv2_shape[1] * conv2_shape[2]
        )

        generator = None

        if seed is not None:
            generator = torch.Generator(
                device="cpu"
            ).manual_seed(int(seed))

        scale = num_classes ** 0.5

        self.feedback_conv1 = (
            torch.randn(
                num_classes,
                self.conv1_flat_size,
                generator=generator,
                dtype=torch.float32,
            ) / scale
        ).to(self.device)

        self.feedback_conv2 = (
            torch.randn(
                num_classes,
                self.conv2_flat_size,
                generator=generator,
                dtype=torch.float32,
            ) / scale
        ).to(self.device)

        self.feedback_dense = (
            torch.randn(
                num_classes,
                dense_size,
                generator=generator,
                dtype=torch.float32,
            ) / scale
        ).to(self.device)

        self.feedback_conv1.requires_grad_(False)
        self.feedback_conv2.requires_grad_(False)
        self.feedback_dense.requires_grad_(False)

    def compute_local_gradients(
        self,
        model: nn.Module,
        x_input: torch.Tensor,
        labels: torch.Tensor,
        forward_intermediates: tuple,
    ) -> dict[str, torch.Tensor]:
        """Compute local DFA gradients without backpropagating through layers."""

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

        if outputs.ndim != 2 or outputs.shape[1] != self.num_classes:
            raise ValueError(
                "Output tensor has an incompatible shape."
            )

        # Ensure model outputs and DFA feedback use the same device.
        if outputs.device.type != self.device.type:
            raise ValueError(
                f"Device mismatch: outputs={outputs.device}, "
                f"DFA={self.device}"
            )

        self.feedback_conv1 = self.feedback_conv1.to(outputs.device)
        self.feedback_conv2 = self.feedback_conv2.to(outputs.device)
        self.feedback_dense = self.feedback_dense.to(outputs.device)

        self.device = outputs.device

        if labels.device != outputs.device:
            raise ValueError(
                "Labels and model outputs must be on the same device."
            )

        if not torch.isfinite(outputs).all():
            raise ValueError(
                "Model outputs contain non-finite values."
            )

        # Output error.
        output_error = CIFARDFAFunction.output_error(
            outputs,
            labels,
        )

        # -----------------------------------------------------
        # Output layer: exact cross-entropy gradient
        # -----------------------------------------------------

        grad_fc2_weight = output_error.T @ dense_act
        grad_fc2_bias = output_error.sum(dim=0)

        # -----------------------------------------------------
        # Dense hidden layer: direct feedback
        # -----------------------------------------------------

        dense_error_signal = output_error @ self.feedback_dense

        dense_delta = (
            dense_error_signal
            * (dense_pre > 0).to(dense_pre.dtype)
        )

        grad_fc1_weight = dense_delta.T @ gap_act
        grad_fc1_bias = dense_delta.sum(dim=0)

        # -----------------------------------------------------
        # Second convolutional layer: direct feedback
        # -----------------------------------------------------

        conv2_error_signal_flat = (
            output_error @ self.feedback_conv2
        )

        conv2_delta = conv2_error_signal_flat.reshape(
            batch_size,
            *self.conv2_shape,
        ) * (conv2_pre > 0).to(conv2_pre.dtype)

        grad_conv2_weight = G.conv2d_weight(
            conv1_act,
            model.conv2.weight.shape,
            conv2_delta,
            stride=model.conv2.stride,
            padding=model.conv2.padding,
        )

        grad_conv2_bias = conv2_delta.sum(
            dim=(0, 2, 3)
        )

        # -----------------------------------------------------
        # First convolutional layer: direct feedback
        # -----------------------------------------------------

        conv1_error_signal_flat = (
            output_error @ self.feedback_conv1
        )

        conv1_delta = conv1_error_signal_flat.reshape(
            batch_size,
            *self.conv1_shape,
        ) * (conv1_pre > 0).to(conv1_pre.dtype)

        grad_conv1_weight = G.conv2d_weight(
            x_input,
            model.conv1.weight.shape,
            conv1_delta,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
        )

        grad_conv1_bias = conv1_delta.sum(
            dim=(0, 2, 3)
        )

        # -----------------------------------------------------
        # Collect gradients
        # -----------------------------------------------------

        gradients = {
            "conv1.weight": grad_conv1_weight,
            "conv1.bias": grad_conv1_bias,
            "conv2.weight": grad_conv2_weight,
            "conv2.bias": grad_conv2_bias,
            "fc1.weight": grad_fc1_weight,
            "fc1.bias": grad_fc1_bias,
            "fc2.weight": grad_fc2_weight,
            "fc2.bias": grad_fc2_bias,
        }

        for name, grad in gradients.items():
            if not torch.isfinite(grad).all():
                raise FloatingPointError(
                    f"Non-finite DFA gradient detected: {name}"
                )

        return gradients

    def apply_gradients(
        self,
        model: nn.Module,
        gradients: dict[str, torch.Tensor],
    ) -> None:
        """Assign manually calculated DFA gradients to model parameters."""

        parameters = {
            "conv1.weight": model.conv1.weight,
            "conv1.bias": model.conv1.bias,
            "conv2.weight": model.conv2.weight,
            "conv2.bias": model.conv2.bias,
            "fc1.weight": model.fc1.weight,
            "fc1.bias": model.fc1.bias,
            "fc2.weight": model.fc2.weight,
            "fc2.bias": model.fc2.bias,
        }

        if gradients.keys() != parameters.keys():
            raise ValueError(
                "DFA gradient keys do not match model parameters."
            )

        for name, parameter in parameters.items():
            gradient = gradients[name]

            if gradient.shape != parameter.shape:
                raise ValueError(
                    f"Gradient shape mismatch for {name}: "
                    f"{gradient.shape} vs {parameter.shape}"
                )

            if gradient.device != parameter.device:
                raise ValueError(
                    f"Gradient device mismatch for {name}."
                )

            parameter.grad = gradient.detach().clone()


class CIFARDFAFunction:
    """Utility functions for computing DFA output error vectors."""

    @staticmethod
    def output_error(
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the mean cross-entropy gradient with respect to logits."""

        if outputs.ndim != 2:
            raise ValueError(
                "Outputs must be a two-dimensional tensor."
            )

        if labels.ndim != 1 or labels.size(0) != outputs.size(0):
            raise ValueError(
                "Labels must have shape [batch_size]."
            )

        if labels.dtype != torch.long:
            labels = labels.long()

        if labels.device != outputs.device:
            raise ValueError(
                "Labels and outputs must share the same device."
            )

        if labels.numel() and (
            labels.min() < 0
            or labels.max() >= outputs.size(1)
        ):
            raise ValueError(
                "Labels contain invalid class indices."
            )

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        targets = torch.zeros_like(probabilities)

        targets.scatter_(
            1,
            labels.unsqueeze(1),
            1.0,
        )

        return (
            probabilities - targets
        ) / labels.size(0)