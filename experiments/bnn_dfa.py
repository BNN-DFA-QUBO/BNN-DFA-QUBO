import torch
import torch.nn as nn
import torch.optim as optim

from models.bnn import BNN
from models.dfa import DFAClassifier, DFAFunction
from utils.data import get_mnist_loaders
from utils.seed import set_seed, get_seed


def train(model, train_loader, optimizer, criterion, dfa):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    device = next(model.parameters()).device

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        x = torch.flatten(images, start_dim=1)

        hidden = model.fc1(x)
        hidden_relu = torch.relu(hidden)

        outputs = model.fc2(hidden_relu)

        loss = criterion(outputs, labels)

        output_error = DFAFunction.output_error(outputs, labels)

        grad_output_weight = output_error.T @ hidden_relu
        grad_output_bias = output_error.sum(dim=0)

        hidden_error = dfa.hidden_gradient(output_error)

        relu_derivative = (hidden > 0).float()
        hidden_error = hidden_error * relu_derivative

        grad_hidden_weight = hidden_error.T @ x
        grad_hidden_bias = hidden_error.sum(dim=0)

        model.fc1.weight.grad = grad_hidden_weight
        model.fc1.bias.grad = grad_hidden_bias

        model.fc2.weight.grad = grad_output_weight
        model.fc2.bias.grad = grad_output_bias

        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def test(model, test_loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            x = torch.flatten(images, start_dim=1)

            hidden = model.fc1(x)
            hidden_relu = torch.relu(hidden)

            outputs = model.fc2(hidden_relu)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def main():
    seed = get_seed()
    set_seed(seed)

    print(f"Using seed: {seed}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_loaders(
        batch_size=64
    )

    model = BNN().to(device)

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
            train_loader,
            optimizer,
            criterion,
            dfa
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

    test_loss, test_accuracy = test(
        model,
        test_loader,
        criterion
    )

    print(
        f"\nFinal Test Loss: {test_loss:.4f} | "
        f"Final Test Accuracy: {test_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()