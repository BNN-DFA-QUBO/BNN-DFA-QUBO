import torch
import torch.nn.functional as F

from models.bnn import BNN
from models.dfa import DFAClassifier, DFAFunction
from utils.data import get_mnist_loaders
from utils.seed import set_seed


def train_dfa(
    model,
    dfa,
    train_loader,
    optimizer,
    criterion,
    device
):
    """
    Train BNN using Direct Feedback Alignment.
    """

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        x = torch.flatten(
            images,
            start_dim=1
        )

        # Hidden layer
        hidden = model.fc1(x)
        hidden_relu = torch.relu(hidden)

        # Output layer
        outputs = model.fc2(hidden_relu)

        loss = criterion(
            outputs,
            labels
        )

        # ------------------------------------------
        # DFA output error
        # ------------------------------------------

        output_error = (
            DFAFunction.output_error(
                outputs,
                labels
            )
        )

        # ------------------------------------------
        # Output layer gradients
        # ------------------------------------------

        grad_output_weight = (
            output_error.T
            @ hidden_relu
        )

        grad_output_bias = (
            output_error.sum(dim=0)
        )

        # ------------------------------------------
        # DFA hidden-layer error
        # ------------------------------------------

        hidden_error = (
            dfa.hidden_gradient(
                output_error
            )
        )

        relu_derivative = (
            hidden > 0
        ).float()

        hidden_error = (
            hidden_error
            * relu_derivative
        )

        # ------------------------------------------
        # Hidden layer gradients
        # ------------------------------------------

        grad_hidden_weight = (
            hidden_error.T
            @ x
        )

        grad_hidden_bias = (
            hidden_error.sum(dim=0)
        )

        # ------------------------------------------
        # Assign gradients
        # ------------------------------------------

        model.fc1.weight.grad = (
            grad_hidden_weight
        )

        model.fc1.bias.grad = (
            grad_hidden_bias
        )

        model.fc2.weight.grad = (
            grad_output_weight
        )

        model.fc2.bias.grad = (
            grad_output_bias
        )

        optimizer.step()

        # ------------------------------------------
        # Statistics
        # ------------------------------------------

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


def collect_hidden_features(
    model,
    loader,
    device
):
    """
    Collect hidden representations.

    Returns:

        H      : hidden features
        labels : corresponding MNIST labels
    """

    model.eval()

    features = []
    labels_all = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            x = torch.flatten(
                images,
                start_dim=1
            )

            hidden = model.fc1(x)

            hidden = torch.relu(hidden)

            features.append(
                hidden.cpu()
            )

            labels_all.append(
                labels.cpu()
            )

    H = torch.cat(features)
    labels = torch.cat(labels_all)

    return H, labels


def fit_least_squares_classifier(
    H,
    labels,
    num_classes=10
):
    """
    Fit a real-valued linear classifier using
    least squares.

    We solve:

        min_W ||H_aug W^T - Y||^2

    where H_aug includes a bias column.
    """

    H = H.float()

    # ------------------------------------------
    # Add bias term
    # ------------------------------------------

    bias = torch.ones(
        H.size(0),
        1,
        dtype=H.dtype
    )

    H_aug = torch.cat(
        [H, bias],
        dim=1
    )

    # ------------------------------------------
    # One-hot targets
    # ------------------------------------------

    Y = F.one_hot(
        labels,
        num_classes=num_classes
    ).float()

    # ------------------------------------------
    # Least-squares solution
    # ------------------------------------------

    solution = torch.linalg.lstsq(
        H_aug,
        Y
    ).solution

    return solution


def evaluate_least_squares(
    model,
    classifier,
    test_loader,
    device
):
    """
    Evaluate the least-squares classifier on
    the test hidden representations.
    """

    model.eval()

    correct = 0
    total = 0

    classifier = classifier.to(device)

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # --------------------------------------
            # Hidden representation
            # --------------------------------------

            x = torch.flatten(
                images,
                start_dim=1
            )

            hidden = model.fc1(x)

            hidden = torch.relu(hidden)

            # --------------------------------------
            # Add bias
            # --------------------------------------

            bias = torch.ones(
                hidden.size(0),
                1,
                device=device
            )

            hidden_aug = torch.cat(
                [hidden, bias],
                dim=1
            )

            # --------------------------------------
            # Linear classifier
            # --------------------------------------

            outputs = (
                hidden_aug @ classifier
            )

            predictions = (
                outputs.argmax(dim=1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100.0 * correct / total


def evaluate_training_fit(
    H,
    labels,
    classifier
):
    """
    Evaluate the least-squares classifier on the
    training representation.
    """

    H = H.float()

    bias = torch.ones(
        H.size(0),
        1,
        dtype=H.dtype
    )

    H_aug = torch.cat(
        [H, bias],
        dim=1
    )

    outputs = (
        H_aug @ classifier
    )

    predictions = (
        outputs.argmax(dim=1)
    )

    accuracy = (
        predictions == labels
    ).float().mean().item() * 100.0

    return accuracy


if __name__ == "__main__":

    # ==================================================
    # Reproducibility
    # ==================================================

    set_seed(42)

    # ==================================================
    # Device
    # ==================================================

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device
    )

    # ==================================================
    # Dataset
    # ==================================================

    train_loader, test_loader = (
        get_mnist_loaders(
            batch_size=64
        )
    )

    # ==================================================
    # Model
    # ==================================================

    model = BNN().to(device)

    dfa = DFAClassifier(
        hidden_size=128,
        output_size=10,
        device=device
    )

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 5

    # ==================================================
    # Train BNN + DFA
    # ==================================================

    print(
        "\nTraining BNN + DFA...\n"
    )

    for epoch in range(epochs):

        train_loss, train_accuracy = (
            train_dfa(
                model,
                dfa,
                train_loader,
                optimizer,
                criterion,
                device
            )
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

    # ==================================================
    # Collect hidden representation
    # ==================================================

    print(
        "\nCollecting hidden representations..."
    )

    H_train, y_train = (
        collect_hidden_features(
            model,
            train_loader,
            device
        )
    )

    print(
        "Training hidden feature shape:",
        H_train.shape
    )

    # ==================================================
    # Fit least-squares classifier
    # ==================================================

    print(
        "\nFitting real-valued least-squares "
        "classifier..."
    )

    classifier = (
        fit_least_squares_classifier(
            H_train,
            y_train,
            num_classes=10
        )
    )

    print(
        "Least-squares classifier shape:",
        classifier.shape
    )

    # ==================================================
    # Training representation accuracy
    # ==================================================

    train_ls_accuracy = (
        evaluate_training_fit(
            H_train,
            y_train,
            classifier
        )
    )

    print(
        f"\nLeast-Squares Training Accuracy: "
        f"{train_ls_accuracy:.2f}%"
    )

    # ==================================================
    # Test accuracy
    # ==================================================

    print(
        "\nEvaluating least-squares classifier..."
    )

    test_ls_accuracy = (
        evaluate_least_squares(
            model,
            classifier,
            test_loader,
            device
        )
    )

    print(
        f"Least-Squares Test Accuracy: "
        f"{test_ls_accuracy:.2f}%"
    )

    # ==================================================
    # Comparison
    # ==================================================

    print(
        "\n=========================================="
    )

    print(
        f"BNN + DFA Test Accuracy: "
        f"93.62%"
    )

    print(
        f"BNN + DFA + Real-Valued LS Head: "
        f"{test_ls_accuracy:.2f}%"
    )

    print(
        f"Difference: "
        f"{test_ls_accuracy - 93.62:+.2f} "
        f"percentage points"
    )

    print(
        "=========================================="
    )