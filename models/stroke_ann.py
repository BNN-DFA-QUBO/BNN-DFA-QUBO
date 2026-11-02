import torch
import torch.nn as nn


class StrokeANN(nn.Module):
    """Standard continuous Artificial Neural Network for Stroke classification.
    
    Uses standard continuous Linear layers with ReLU activation and backpropagation,
    directly corresponding to MNIST's BaselineANN.
    """
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x.squeeze(1)

