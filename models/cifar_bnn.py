"""CIFAR-10 Binary Neural Network (BNN) with convolutional architecture.

Architecture:
    Input: [B, 3, 32, 32]
      -> BinaryConv2d(3, 32, kernel=3, stride=2, padding=1)
      -> ReLU -> [B, 32, 16, 16] (8,192 hidden activation features per sample)
      -> BinaryConv2d(32, 64, kernel=3, stride=2, padding=1)
      -> ReLU -> [B, 64, 8, 8] (4,096 hidden activation features per sample)
      -> Global Average Pooling (AdaptiveAvgPool2d(1, 1)) -> [B, 64]
      -> BinaryLinear(64, 128)
      -> ReLU -> [B, 128] (128-dimensional frozen representation)
      -> BinaryLinear(128, 10) -> [B, 10] class logits

Weight Binarization:
    - Real-valued latent weights are maintained for optimization.
    - Forward pass uses scaled binary weights: W_bin = sign(W) * mean(|W|).
    - Straight-Through Estimator (STE) allows gradients to update latent weights:
      W_eff = W_bin.detach() - W.detach() + W.
    - Activations remain continuous (standard ReLU) to preserve representational capacity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bnn import BinaryLinear


class BinaryConv2d(nn.Conv2d):
    """2D Convolutional layer with binary weights during the forward pass.

    Maintains real-valued latent weights updated via STE, scaled by the mean
    absolute weight value, matching BinaryLinear convention.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Binarize weights to {-1, +1}
        binary_weight = self.weight.sign()

        # Scale binary weights using mean absolute weight
        scale = self.weight.abs().mean()
        binary_weight = binary_weight * scale

        # Straight-Through Estimator
        binary_weight = (
            binary_weight.detach()
            - self.weight.detach()
            + self.weight
        )

        return F.conv2d(
            x,
            binary_weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class CIFAR_BNN(nn.Module):
    """Convolutional BNN backbone for CIFAR-10 producing a 128-d representation."""

    def __init__(self):
        super().__init__()
        self.conv1 = BinaryConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = BinaryConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = BinaryLinear(64, 128)
        self.fc2 = BinaryLinear(128, 10)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the 128-dimensional representation (post-fc1 ReLU).

        Parameters
        ----------
        x: [B, 3, 32, 32] input batch

        Returns
        -------
        [B, 128] frozen feature tensor
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass returning 10-class output logits."""
        h = self.extract_features(x)
        return self.fc2(h)

    def forward_with_intermediates(self, x: torch.Tensor):
        """Forward pass recording pre- and post-activation tensors for DFA updates.

        Returns
        -------
        tuple: (
            outputs,       # [B, 10]
            conv1_pre,     # [B, 32, 16, 16]
            conv1_act,     # [B, 32, 16, 16]
            conv2_pre,     # [B, 64, 8, 8]
            conv2_act,     # [B, 64, 8, 8]
            gap_act,       # [B, 64]
            dense_pre,     # [B, 128]
            dense_act      # [B, 128]
        )
        """
        conv1_pre = self.conv1(x)
        conv1_act = torch.relu(conv1_pre)

        conv2_pre = self.conv2(conv1_act)
        conv2_act = torch.relu(conv2_pre)

        gap_out = self.gap(conv2_act)
        gap_act = torch.flatten(gap_out, start_dim=1)

        dense_pre = self.fc1(gap_act)
        dense_act = torch.relu(dense_pre)

        outputs = self.fc2(dense_act)

        return (
            outputs,
            conv1_pre,
            conv1_act,
            conv2_pre,
            conv2_act,
            gap_act,
            dense_pre,
            dense_act,
        )


if __name__ == "__main__":
    model = CIFAR_BNN()
    dummy_x = torch.randn(4, 3, 32, 32)
    feats = model.extract_features(dummy_x)
    logits = model(dummy_x)
    print("CIFAR_BNN Model Summary:")
    print("Features shape:", feats.shape)
    print("Logits shape:", logits.shape)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
