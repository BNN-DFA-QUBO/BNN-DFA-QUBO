"""CIFAR-10 Binary Neural Network (BNN) with convolutional architecture.

Architecture:
    Input: [B, 3, 32, 32]
      -> BinaryConv2d(3, 32, kernel=3, stride=2, padding=1)
      -> ReLU -> [B, 32, 16, 16]
      -> BinaryConv2d(32, 64, kernel=3, stride=2, padding=1)
      -> ReLU -> [B, 64, 8, 8]
      -> Adaptive Average Pooling (4x4) -> [B, 1024]
      -> BinaryLinear(1024, 256)
      -> ReLU -> [B, 128]
      -> BinaryLinear(256, 10)
      -> [B, 10] class logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bnn import BinaryLinear


class BinaryConv2d(nn.Conv2d):
    """Convolutional layer with scaled binary weights and STE."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binary_weight = self.weight.sign()

        scale = self.weight.abs().mean()
        binary_weight = binary_weight * scale

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

    def __init__(self):
        super().__init__()

        self.conv1 = BinaryConv2d(
            3, 32, kernel_size=3, stride=2, padding=1
        )

        self.conv2 = BinaryConv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )

        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = BinaryLinear(1024, 256)
        self.fc2 = BinaryLinear(256, 10)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.fc1(x))

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h = self.extract_features(x)
        return self.fc2(h)

    def forward_with_intermediates(self, x: torch.Tensor):

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

    outputs = model.forward_with_intermediates(dummy_x)

    print("CIFAR_BNN Model Summary:")
    print("Logits shape:", outputs[0].shape)
    print("Conv1 activation:", outputs[2].shape)
    print("Conv2 activation:", outputs[4].shape)
    print("GAP output:", outputs[5].shape)
    print("Dense representation:", outputs[7].shape)
    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )