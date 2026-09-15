from utils.seed import set_seed
import torch
import torch.nn as nn
import torch.optim as optim

from models.bnn import BNN
from models.dfa import DFAClassifier, DFAFunction
from utils.data import get_mnist_loaders


def train(model, dfa, train_loader, optimizer, criterion, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        x = torch.flatten(images, start_dim=1)

        hidden = model.fc1(x)
        hidden_relu = torch.relu(hidden)

        outputs = model.fc2(hidden_relu)

        # Calculate loss for monitoring
        loss = criterion(outputs, labels)

        # ------------------------------------------------
        # DFA: output error
        # ------------------------------------------------
        output_error = DFAFunction.output_error(
            outputs,
            labels
        )

        # ------------------------------------------------
        # Output layer gradient
        # ------------------------------------------------
        grad_output_weight = output_error.T @ hidden_relu

        grad_output_bias = output_error.sum(dim=0)

        # ------------------------------------------------
        # DFA hidden-layer gradient
        # ------------------------------------------------
        hidden_error = dfa.hidden_gradient(output_error)

        # ReLU derivative
        relu_derivative = (hidden > 0).float()

        hidden_error = hidden_error * relu_derivative

        # Input gradient for fc1
        grad_hidden_weight = hidden_error.T @ x

        grad_hidden_bias = hidden_error.sum(dim=0)

        # ------------------------------------------------
        # Assign gradients manually
        # ------------------------------------------------
        model.fc1.weight.grad = grad_hidden_weight
        model.fc1.bias.grad = grad_hidden_bias

        model.fc2.weight.grad = grad_output_weight
        model.fc2.bias.grad = grad_output_bias

        # Update parameters
        optimizer.step()

        # Statistics
        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = total_loss / len(train_loader)

    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def test(model, test_loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / len(test_loader)

    accuracy = 100.0 * correct / total

    return average_loss, accuracy


if __name__ == "__main__":

    set_seed(42)
    
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    train_loader, test_loader = get_mnist_loaders(
        batch_size=64
    )

    model = BNN().to(device)

    # DFA feedback matrix
    dfa = DFAClassifier(
        hidden_size=128,
        output_size=10,
        device=device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 16

    print("\nTraining BNN + DFA...\n")

    for epoch in range(epochs):

        train_loss, train_accuracy = train(
            model,
            dfa,
            train_loader,
            optimizer,
            criterion,
            device
        )

        test_loss, test_accuracy = test(
            model,
            test_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_accuracy:.2f}%"
        )