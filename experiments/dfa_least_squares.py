import torch
import torch.nn.functional as F

from models.bnn import BNN
from models.dfa import DFAClassifier, DFAFunction
from utils.data import get_mnist_loaders
from utils.seed import set_seed


# ============================================================
# BNN + DFA TRAINING
# ============================================================

def train_dfa(
    model,
    dfa,
    train_loader,
    optimizer,
    criterion,
    device
):
    """
    Train the BNN using Direct Feedback Alignment (DFA).

    The output error is propagated directly to the hidden
    layer through a fixed random feedback matrix.
    """

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        x = torch.flatten(images, start_dim=1)

        # Hidden layer
        hidden = model.fc1(x)
        hidden_relu = torch.relu(hidden)

        # Output layer
        outputs = model.fc2(hidden_relu)

        # Loss for monitoring
        loss = criterion(outputs, labels)

        # ----------------------------------------------------
        # DFA output error
        # ----------------------------------------------------

        output_error = DFAFunction.output_error(
            outputs,
            labels
        )

        # ----------------------------------------------------
        # Output layer gradients
        # ----------------------------------------------------

        grad_output_weight = (
            output_error.T @ hidden_relu
        )

        grad_output_bias = (
            output_error.sum(dim=0)
        )

        # ----------------------------------------------------
        # Direct Feedback Alignment
        # ----------------------------------------------------

        hidden_error = dfa.hidden_gradient(
            output_error
        )

        # ReLU derivative
        relu_derivative = (
            hidden > 0
        ).float()

        hidden_error = (
            hidden_error * relu_derivative
        )

        # ----------------------------------------------------
        # Hidden layer gradients
        # ----------------------------------------------------

        grad_hidden_weight = (
            hidden_error.T @ x
        )

        grad_hidden_bias = (
            hidden_error.sum(dim=0)
        )

        # ----------------------------------------------------
        # Manually assign gradients
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

        # ----------------------------------------------------
        # Update parameters
        # ----------------------------------------------------

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

    average_loss = (
        total_loss / len(train_loader)
    )

    accuracy = (
        100.0 * correct / total
    )

    return average_loss, accuracy


# ============================================================
# EVALUATE ORIGINAL BNN + DFA MODEL
# ============================================================

def evaluate_dfa_model(
    model,
    test_loader,
    criterion,
    device
):
    """
    Evaluate the original trained BNN + DFA classifier.

    This is important because the LS experiment must compare
    its new head against the actual BNN + DFA model produced
    during the same run.
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass through original BNN
            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        total_loss / len(test_loader)
    )

    accuracy = (
        100.0 * correct / total
    )

    return average_loss, accuracy


# ============================================================
# COLLECT HIDDEN REPRESENTATIONS
# ============================================================

def collect_hidden_features(
    model,
    loader,
    device
):
    """
    Collect hidden representations from the trained BNN.

    Returns
    -------
    H:
        Hidden feature matrix.

    labels:
        Corresponding labels.
    """

    model.eval()

    features = []
    labels_all = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            # Flatten input
            x = torch.flatten(
                images,
                start_dim=1
            )

            # Hidden representation
            hidden = model.fc1(x)

            hidden = torch.relu(hidden)

            # Move to CPU to reduce MPS memory usage
            features.append(
                hidden.cpu()
            )

            labels_all.append(
                labels.cpu()
            )

    H = torch.cat(features)

    labels = torch.cat(
        labels_all
    )

    return H, labels


# ============================================================
# FIT LEAST-SQUARES CLASSIFIER
# ============================================================

def fit_least_squares_classifier(
    H,
    labels,
    num_classes=10
):
    """
    Fit a real-valued linear classifier using least squares.

    We solve:

        min_W ||H_aug W - Y||²

    where:

        H_aug = [H, 1]

    The final column allows the classifier to learn a bias.
    """

    H = H.float()

    # --------------------------------------------------------
    # Add bias column
    # --------------------------------------------------------

    bias = torch.ones(
        H.size(0),
        1,
        dtype=H.dtype
    )

    H_aug = torch.cat(
        [H, bias],
        dim=1
    )

    # --------------------------------------------------------
    # One-hot encode targets
    # --------------------------------------------------------

    Y = F.one_hot(
        labels,
        num_classes=num_classes
    ).float()

    # --------------------------------------------------------
    # Least-squares solution
    # --------------------------------------------------------

    solution = torch.linalg.lstsq(
        H_aug,
        Y
    ).solution

    return solution


# ============================================================
# EVALUATE LEAST-SQUARES CLASSIFIER
# ============================================================

def evaluate_least_squares(
    model,
    classifier,
    test_loader,
    device
):
    """
    Evaluate the least-squares classifier on the
    test hidden representations.
    """

    model.eval()

    correct = 0
    total = 0

    classifier = classifier.to(device)

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # ------------------------------------------------
            # Hidden representation
            # ------------------------------------------------

            x = torch.flatten(
                images,
                start_dim=1
            )

            hidden = model.fc1(x)

            hidden = torch.relu(hidden)

            # ------------------------------------------------
            # Add bias
            # ------------------------------------------------

            bias = torch.ones(
                hidden.size(0),
                1,
                device=device
            )

            hidden_aug = torch.cat(
                [hidden, bias],
                dim=1
            )

            # ------------------------------------------------
            # LS classifier
            # ------------------------------------------------

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

    accuracy = (
        100.0 * correct / total
    )

    return accuracy


# ============================================================
# TRAINING-SET FIT OF LS CLASSIFIER
# ============================================================

def evaluate_training_fit(
    H,
    labels,
    classifier
):
    """
    Evaluate the LS classifier on the same training
    representation used to fit it.
    """

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

    # Classifier output
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


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Reproducibility
    # ========================================================

    set_seed(42)

    # ========================================================
    # Device
    # ========================================================

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

    # ========================================================
    # Dataset
    # ========================================================

    train_loader, test_loader = (
        get_mnist_loaders(
            batch_size=64
        )
    )

    # ========================================================
    # Model
    # ========================================================

    model = BNN().to(device)

    # DFA feedback matrix
    dfa = DFAClassifier(
        hidden_size=128,
        output_size=10,
        device=device
    )

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Number of epochs
    epochs = 5

    # ========================================================
    # Train BNN + DFA
    # ========================================================

    print(
        "\nTraining BNN + DFA...\n"
    )

    for epoch in range(epochs):

        train_loss, train_accuracy = (
            train_dfa(
                model=model,
                dfa=dfa,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device
            )
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

    # ========================================================
    # Evaluate ORIGINAL BNN + DFA
    # ========================================================

    print(
        "\nEvaluating original BNN + DFA model..."
    )

    dfa_test_loss, dfa_test_accuracy = (
        evaluate_dfa_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )
    )

    print(
        f"BNN + DFA Test Loss: "
        f"{dfa_test_loss:.4f}"
    )

    print(
        f"BNN + DFA Test Accuracy: "
        f"{dfa_test_accuracy:.2f}%"
    )

    # ========================================================
    # Collect hidden representations
    # ========================================================

    print(
        "\nCollecting hidden representations..."
    )

    H_train, y_train = (
        collect_hidden_features(
            model=model,
            loader=train_loader,
            device=device
        )
    )

    print(
        "Training hidden feature shape:",
        H_train.shape
    )

    # ========================================================
    # Fit LS classifier
    # ========================================================

    print(
        "\nFitting real-valued least-squares classifier..."
    )

    classifier = (
        fit_least_squares_classifier(
            H=H_train,
            labels=y_train,
            num_classes=10
        )
    )

    print(
        "Least-squares classifier shape:",
        classifier.shape
    )

    # ========================================================
    # Training accuracy of LS head
    # ========================================================

    train_ls_accuracy = (
        evaluate_training_fit(
            H=H_train,
            labels=y_train,
            classifier=classifier
        )
    )

    print(
        f"\nLeast-Squares Training Accuracy: "
        f"{train_ls_accuracy:.2f}%"
    )

    # ========================================================
    # Test accuracy of LS head
    # ========================================================

    print(
        "\nEvaluating least-squares classifier..."
    )

    test_ls_accuracy = (
        evaluate_least_squares(
            model=model,
            classifier=classifier,
            test_loader=test_loader,
            device=device
        )
    )

    print(
        f"Least-Squares Test Accuracy: "
        f"{test_ls_accuracy:.2f}%"
    )

    # ========================================================
    # Final comparison
    # ========================================================

    difference = (
        test_ls_accuracy
        - dfa_test_accuracy
    )

    print(
        "\n=========================================="
    )

    print(
        f"BNN + DFA: "
        f"{dfa_test_accuracy:.2f}%"
    )

    print(
        f"BNN + DFA + Real-Valued LS Head: "
        f"{test_ls_accuracy:.2f}%"
    )

    print(
        f"Difference: "
        f"{difference:+.2f} percentage points"
    )

    print(
        "=========================================="
    )