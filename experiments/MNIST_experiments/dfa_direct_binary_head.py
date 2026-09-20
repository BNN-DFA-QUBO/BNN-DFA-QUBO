import torch
import torch.nn.functional as F

from models.bnn import BNN
from models.dfa import DFAClassifier, DFAFunction
from utils.MNIST.data import get_mnist_loaders
from utils.MNIST.seed import set_seed, get_seed

from experiments.MNIST_experiments.results_utils import (
    append_result,
    Timer,
    count_parameters,
)


def train_dfa(model, train_loader, optimizer, dfa, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        x = torch.flatten(images, start_dim=1)

        hidden = model.fc1(x)
        hidden = torch.relu(hidden)

        outputs = model.fc2(hidden)

        loss = F.cross_entropy(outputs, labels)

        # Output error
        output_error = DFAFunction.output_error(
            outputs,
            labels
        )

        # Direct feedback to hidden layer
        hidden_error = dfa.hidden_gradient(
            output_error
        )

        # Hidden pre-activation
        hidden_pre = model.fc1(x)

        # ReLU derivative
        hidden_grad = hidden_error * (
            hidden_pre > 0
        ).float()

        # First layer gradients
        grad_w1 = hidden_grad.t() @ x
        grad_b1 = hidden_grad.sum(dim=0)

        # Second layer gradients
        grad_w2 = output_error.t() @ hidden
        grad_b2 = output_error.sum(dim=0)

        model.fc1.weight.grad = grad_w1
        model.fc1.bias.grad = grad_b1

        model.fc2.weight.grad = grad_w2
        model.fc2.bias.grad = grad_b2

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / len(train_loader),
        100.0 * correct / total
    )


@torch.no_grad()
def collect_hidden_features(
    model,
    loader,
    device
):

    model.eval()

    features = []
    labels_all = []

    for images, labels in loader:

        images = images.to(device)

        x = torch.flatten(
            images,
            start_dim=1
        )

        hidden = torch.relu(
            model.fc1(x)
        )

        features.append(
            hidden.cpu()
        )

        labels_all.append(
            labels.cpu()
        )

    return (
        torch.cat(features),
        torch.cat(labels_all)
    )


def optimize_binary_head(
    H,
    labels,
    num_classes=10,
    steps=200,
    lr=0.05
):

    H = H.float()

    hidden_size = H.shape[1]

    # One-hot targets
    Y = F.one_hot(
        labels,
        num_classes=num_classes
    ).float()

    # Latent real-valued parameters.
    # These are NOT the final classifier weights.
    latent = torch.randn(
        hidden_size,
        num_classes,
        dtype=torch.float32,
        requires_grad=True
    )

    bias = torch.zeros(
        num_classes,
        dtype=torch.float32,
        requires_grad=True
    )

    optimizer = torch.optim.Adam(
        [latent, bias],
        lr=lr
    )

    for step in range(steps):

        optimizer.zero_grad()

        # Continuous relaxation of binary weights.
        #
        # tanh approaches {-1,+1}
        # as the latent values become large.
        W = torch.tanh(latent)

        logits = H @ W + bias

        loss = F.cross_entropy(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

    # Final hard binary projection
    with torch.no_grad():

        W_binary = torch.where(
            latent >= 0,
            torch.ones_like(latent),
            -torch.ones_like(latent)
        )

        logits = H @ W_binary + bias

        predictions = logits.argmax(dim=1)

        train_accuracy = (
            100.0 *
            (predictions == labels)
            .float()
            .mean()
            .item()
        )

    return (
        W_binary,
        bias.detach(),
        train_accuracy
    )


@torch.no_grad()
def evaluate_binary_head(
    model,
    binary_weights,
    bias,
    test_loader,
    device
):

    model.eval()

    W = binary_weights.to(device)
    b = bias.to(device)

    correct = 0
    total = 0

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        x = torch.flatten(
            images,
            start_dim=1
        )

        hidden = torch.relu(
            model.fc1(x)
        )

        outputs = hidden @ W + b

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return 100.0 * correct / total



    


if __name__ == "__main__":

    seed = get_seed()
    set_seed(seed)

    print(f"Using seed: {seed}")

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    train_loader, test_loader = get_mnist_loaders(
        batch_size=64
    )

    model = BNN().to(device)

    dfa = DFAClassifier(
        hidden_size=128,
        output_size=10,
        device=device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    print("\nTraining BNN + DFA...")

    # Time the complete 16-epoch training process
    with Timer() as training_timer:

        for epoch in range(16):

            train_loss, train_accuracy = train_dfa(
                model,
                train_loader,
                optimizer,
                dfa,
                device
            )

            print(
                f"Epoch {epoch + 1}/16 | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: "
                f"{train_accuracy:.2f}%"
            )

    print(
        f"\nTraining time: "
        f"{training_timer.seconds:.2f} seconds"
    )

    print("\nCollecting hidden representations...")

    H_train, y_train = collect_hidden_features(
        model,
        train_loader,
        device
    )

    print(
        "Hidden feature shape:",
        H_train.shape
    )

    print(
        "\nDirectly optimizing binary "
        "classifier head..."
    )

    (
        binary_weights,
        bias,
        head_train_accuracy
    ) = optimize_binary_head(
        H_train,
        y_train,
        num_classes=10,
        steps=200,
        lr=0.05
    )

    print(
        "\nBinary weight shape:",
        binary_weights.shape
    )

    print(
        "Unique binary weight values:",
        torch.unique(binary_weights)
    )

    print(
        f"\nDirect Binary Head "
        f"Training Accuracy: "
        f"{head_train_accuracy:.2f}%"
    )

    print(
        "\nEvaluating direct binary "
        "classifier head..."
    )

    test_accuracy = evaluate_binary_head(
        model,
        binary_weights,
        bias,
        test_loader,
        device
    )

    print("\n==========================================")

    print(
        f"BNN + DFA + Direct Binary Head "
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print("==========================================")

    # Record result for Results & Analysis
    append_result(
        experiment="dfa_direct_binary_head",
        method="BNN + DFA + Direct Binary Head",
        seed=seed,
        test_accuracy=test_accuracy,
        test_loss=None,
        training_time_sec=training_timer.seconds,
        parameter_count=count_parameters(model),
    )