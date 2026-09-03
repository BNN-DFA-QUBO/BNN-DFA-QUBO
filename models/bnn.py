import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLinear(nn.Linear):
    """
    Linear layer with binary weights during the forward pass.

    Real-valued latent weights are maintained for optimization.
    The forward pass uses scaled binary weights, while the
    Straight-Through Estimator (STE) allows gradients to update
    the underlying real-valued weights.
    """

    def forward(self, x):

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

        return F.linear(
            x,
            binary_weight,
            self.bias
        )


class BNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = BinaryLinear(28 * 28, 128)
        self.fc2 = BinaryLinear(128, 10)

    def forward(self, x):

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)

        x = torch.relu(x)

        x = self.fc2(x)

        return x


if __name__ == "__main__":

    model = BNN()

    x = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])

    criterion = nn.CrossEntropyLoss()

    outputs = model(x)

    loss = criterion(outputs, labels)

    loss.backward()

    print(model)

    print(
        "Unique binary weight values:",
        torch.unique(model.fc1.weight.sign())
    )

    print("Loss:", loss.item())

    print(
        "Gradient exists:",
        model.fc1.weight.grad is not None
    )

    print(
        "Gradient mean:",
        model.fc1.weight.grad.abs().mean().item()
    )