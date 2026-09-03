import torch
import torch.nn.functional as F

from models.bnn import BNN
from models.dfa import DFAClassifier, DFAFunction
from utils.data import get_mnist_loaders
from utils.seed import set_seed


# ============================================================
# DFA TRAINING
# ============================================================

def train_dfa(
    model,
    dfa,
    train_loader,
    optimizer,
    criterion,
    device
):

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

        # ----------------------------------------------------
        # DFA output error
        # ----------------------------------------------------

        output_error = (
            DFAFunction.output_error(
                outputs,
                labels
            )
        )

        # ----------------------------------------------------
        # Output layer gradients
        # ----------------------------------------------------

        grad_output_weight = (
            output_error.T
            @ hidden_relu
        )

        grad_output_bias = (
            output_error.sum(dim=0)
        )

        # ----------------------------------------------------
        # DFA hidden error
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Hidden layer gradients
        # ----------------------------------------------------

        grad_hidden_weight = (
            hidden_error.T
            @ x
        )

        grad_hidden_bias = (
            hidden_error.sum(dim=0)
        )

        # ----------------------------------------------------
        # Assign gradients
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

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


# ============================================================
# COLLECT HIDDEN REPRESENTATIONS
# ============================================================

def collect_hidden_features(
    model,
    loader,
    device
):

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


# ============================================================
# LEAST-SQUARES CLASSIFIER
# ============================================================

def fit_least_squares(
    H,
    labels,
    num_classes=10
):

    H = H.float()

    # Add bias
    bias = torch.ones(
        H.size(0),
        1,
        dtype=H.dtype
    )

    H_aug = torch.cat(
        [H, bias],
        dim=1
    )

    # One-hot targets
    Y = F.one_hot(
        labels,
        num_classes=num_classes
    ).float()

    # Solve:
    #
    # H_aug W = Y
    #
    # W shape = 129 x 10

    W = torch.linalg.lstsq(
        H_aug,
        Y
    ).solution

    return W


# ============================================================
# BINARY CONVERSION
# ============================================================

def binarize_classifier(
    classifier
):

    # classifier:
    #
    # 129 x 10
    #
    # First 128 rows = weights
    # Last row = bias

    real_weights = classifier[:-1, :]

    real_bias = classifier[-1, :]

    # Binary weights
    binary_weights = real_weights.sign()

    # --------------------------------------------------------
    # Calculate optimal scale for each class
    # --------------------------------------------------------

    scales = []

    for class_id in range(10):

        w = binary_weights[:, class_id]

        prediction = (
            H_global @ w
        )

        target = (
            Y_global[:, class_id]
        )

        denominator = (
            prediction @ prediction
        )

        if denominator > 1e-12:

            scale = (
                prediction @ target
            ) / denominator

        else:

            scale = torch.tensor(
                1.0,
                dtype=prediction.dtype
            )

        # Keep positive scale by flipping direction
        if scale < 0:

            binary_weights[:, class_id] *= -1

            scale = -scale

        scales.append(scale)

    scales = torch.stack(scales)

    return (
        binary_weights,
        scales,
        real_bias
    )


# ============================================================
# EVALUATE BINARY HEAD
# ============================================================

def evaluate_binary_head(
    model,
    binary_weights,
    scales,
    bias,
    test_loader,
    device
):

    model.eval()

    binary_weights = (
        binary_weights
        .to(device)
    )

    scales = (
        scales
        .to(device)
    )

    bias = (
        bias
        .to(device)
    )

    # Apply class-specific scales
    effective_weights = (
        binary_weights
        * scales.unsqueeze(0)
    )

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            x = torch.flatten(
                images,
                start_dim=1
            )

            hidden = model.fc1(x)

            hidden = torch.relu(hidden)

            outputs = (
                hidden
                @ effective_weights
            )

            outputs = (
                outputs + bias
            )

            predictions = (
                outputs.argmax(dim=1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        100.0 * correct / total
    )


# ============================================================
# BINARY HEAD TRAINING FIT
# ============================================================

def evaluate_binary_training_fit(
    H,
    labels,
    binary_weights,
    scales,
    bias
):

    effective_weights = (
        binary_weights
        * scales.unsqueeze(0)
    )

    outputs = (
        H @ effective_weights
    )

    outputs = (
        outputs + bias
    )

    predictions = (
        outputs.argmax(dim=1)
    )

    accuracy = (
        predictions == labels
    ).float().mean().item() * 100.0

    return accuracy


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(42)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_loader, test_loader = (
        get_mnist_loaders(
            batch_size=64
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Train DFA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Collect hidden representation
    # --------------------------------------------------------

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
        "Hidden feature shape:",
        H_train.shape
    )

    # --------------------------------------------------------
    # Least-squares head
    # --------------------------------------------------------

    print(
        "\nFitting real-valued least-squares head..."
    )

    classifier = fit_least_squares(
        H_train,
        y_train
    )

    print(
        "LS classifier shape:",
        classifier.shape
    )

    # --------------------------------------------------------
    # Global variables used by binary conversion
    # --------------------------------------------------------

    H_global = H_train

    Y_global = F.one_hot(
        y_train,
        num_classes=10
    ).float()

    # --------------------------------------------------------
    # Binarize LS classifier
    # --------------------------------------------------------

    print(
        "\nBinarizing least-squares classifier..."
    )

    (
        binary_weights,
        scales,
        bias
    ) = binarize_classifier(
        classifier
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
        "\nClass-specific scales:"
    )

    for class_id, scale in enumerate(scales):

        print(
            f"  Class {class_id}: "
            f"{scale.item():.6f}"
        )

    # --------------------------------------------------------
    # Training fit
    # --------------------------------------------------------

    binary_train_accuracy = (
        evaluate_binary_training_fit(
            H_train,
            y_train,
            binary_weights,
            scales,
            bias
        )
    )

    print(
        f"\nBinary Head Training Accuracy: "
        f"{binary_train_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print(
        "\nEvaluating binary classifier head..."
    )

    binary_test_accuracy = (
        evaluate_binary_head(
            model,
            binary_weights,
            scales,
            bias,
            test_loader,
            device
        )
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "BNN + DFA + Real-Valued LS Head: "
        "94.05%"
    )

    print(
        f"BNN + DFA + Binarized LS Head: "
        f"{binary_test_accuracy:.2f}%"
    )

    print(
        f"Difference from LS: "
        f"{binary_test_accuracy - 94.05:+.2f} "
        f"percentage points"
    )

    print(
        "=========================================="
    )