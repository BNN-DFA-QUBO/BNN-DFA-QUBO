import torch
import torch.nn.functional as F

from models.bnn import BNN
from models.dfa import DFAClassifier, DFAFunction
from utils.data import get_mnist_loaders
from utils.seed import set_seed, get_seed


def binary_ste(w):
    """
    Hard binary weights in the forward pass:
        {-1, +1}

    Straight-Through Estimator in the backward pass:
        gradient flows through the real-valued weights.
    """

    binary = torch.where(
        w >= 0,
        torch.ones_like(w),
        -torch.ones_like(w)
    )

    return binary.detach() - w.detach() + w


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

        # Hidden layer
        hidden = model.fc1(x)
        hidden = torch.relu(hidden)

        # Output layer
        outputs = model.fc2(hidden)

        loss = F.cross_entropy(outputs, labels)

        # DFA output error
        output_error = DFAFunction.output_error(
            outputs,
            labels
        )

        # Direct feedback
        hidden_error = dfa.hidden_gradient(
            output_error
        )

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
def collect_hidden_features(model, loader, device):

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

        features.append(hidden.cpu())
        labels_all.append(labels.cpu())

    return (
        torch.cat(features),
        torch.cat(labels_all)
    )


def train_binary_head_ste(
    H,
    labels,
    num_classes=10,
    epochs=50,
    lr=0.01
):
    """
    Directly trains a binary classifier head.

    Forward:
        W_binary ∈ {-1,+1}

    Backward:
        STE through the latent real-valued weights.
    """

    H = H.float()

    hidden_size = H.shape[1]

    # Latent real-valued parameters.
    latent_weights = torch.randn(
        hidden_size,
        num_classes,
        dtype=torch.float32
    ) * 0.01

    latent_weights.requires_grad_()

    bias = torch.zeros(
        num_classes,
        dtype=torch.float32,
        requires_grad=True
    )

    optimizer = torch.optim.Adam(
        [latent_weights, bias],
        lr=lr
    )

    for epoch in range(epochs):

        optimizer.zero_grad()

        # HARD binary weights during forward pass
        W_binary = binary_ste(
            latent_weights
        )

        logits = H @ W_binary + bias

        loss = F.cross_entropy(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
        ):
            predictions = logits.argmax(dim=1)

            accuracy = (
                100.0 *
                (predictions == labels)
                .float()
                .mean()
                .item()
            )

            print(
                f"Binary Head Epoch "
                f"{epoch + 1}/{epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {accuracy:.2f}%"
            )

    # Final hard binary weights
    with torch.no_grad():

        W_binary = torch.where(
            latent_weights >= 0,
            torch.ones_like(latent_weights),
            -torch.ones_like(latent_weights)
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

    train_loader, test_loader = (
        get_mnist_loaders(batch_size=64)
    )

    # ------------------------------------------
    # STEP 1: Train BNN + DFA
    # ------------------------------------------

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

    # ------------------------------------------
    # STEP 2: Collect DFA representation
    # ------------------------------------------

    print(
        "\nCollecting hidden representations..."
    )

    H_train, y_train = collect_hidden_features(
        model,
        train_loader,
        device
    )

    print(
        "Hidden feature shape:",
        H_train.shape
    )

    # ------------------------------------------
    # STEP 3: Direct binary optimization
    # ------------------------------------------

    print(
        "\nTraining binary classifier "
        "using STE..."
    )

    (
        binary_weights,
        bias,
        train_accuracy
    ) = train_binary_head_ste(
        H_train,
        y_train,
        num_classes=10,
        epochs=50,
        lr=0.01
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
        f"\nFinal Binary Head "
        f"Training Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    # ------------------------------------------
    # STEP 4: Test
    # ------------------------------------------

    print(
        "\nEvaluating STE binary classifier..."
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
        f"BNN + DFA + STE Binary Head "
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print("==========================================")