"""Bank dataset neural network models (ANN and BNN)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bnn import BinaryLinear


class BankANN(nn.Module):
    """Continuous full-precision (FP32) feedforward baseline network for Bank dataset."""

    def __init__(self, input_size: int = 66, hidden_size: int = 64, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return F.relu(self.fc1(x))


class BankBNN(nn.Module):
    """Binarized Neural Network (BNN) for Bank dataset with Straight-Through Estimator."""

    def __init__(self, input_size: int = 66, hidden_size: int = 64, num_classes: int = 2):
        super().__init__()
        self.fc1 = BinaryLinear(input_size, hidden_size)
        self.fc2 = BinaryLinear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return F.relu(self.fc1(x))

