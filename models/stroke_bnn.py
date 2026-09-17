import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLinear(nn.Linear):
    def forward(self, x):
        binary_weight = self.weight.sign()
        scale = self.weight.abs().mean()
        binary_weight = binary_weight * scale

        binary_weight = (
            binary_weight.detach()
            - self.weight.detach()
            + self.weight
        )

        return F.linear(x, binary_weight, self.bias)


class StrokeBNN(nn.Module):
    def __init__(self, input_size=21, hidden_size=64):
        super().__init__()

        self.fc1 = BinaryLinear(input_size, hidden_size)
        self.fc2 = BinaryLinear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x.squeeze(1)