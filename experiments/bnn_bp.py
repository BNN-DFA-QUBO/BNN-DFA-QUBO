from utils.seed import set_seed
import torch
import torch.nn as nn
import torch.optim as optim

from models.bnn import BNN
from utils.data import get_mnist_loaders


def train(model, train_loader, optimizer, criterion, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
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

            correct += (predictions == labels).sum().item()
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

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 16

    print("\nTraining BNN + Backpropagation...\n")

    for epoch in range(epochs):

        train_loss, train_accuracy = train(
            model,
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