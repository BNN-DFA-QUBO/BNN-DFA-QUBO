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


    append_result(
        experiment="dfa_least_squares",
        method="BNN + DFA + Real-Valued LS Head",
        seed=seed,
        test_accuracy=test_accuracy,
        test_loss=test_loss if "test_loss" in locals() else None,
        training_time_sec=training_timer.seconds if "training_timer" in locals() else None,
        parameter_count=count_parameters(model),
    )


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

    train_loader, test_loader = get_mnist_loaders(batch_size=64)

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

    epochs = 16

    print("\nTraining BNN + DFA...\n")

    with Timer() as training_timer:
        for epoch in range(epochs):
            train_loss, train_accuracy = train_dfa(
                model=model,
                dfa=dfa,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device
            )

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.2f}%"
            )

    dfa_test_loss, dfa_test_accuracy = evaluate_dfa_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(
        f"\nBNN + DFA Test Loss: {dfa_test_loss:.4f}"
    )
    print(
        f"BNN + DFA Test Accuracy: {dfa_test_accuracy:.2f}%"
    )

    print("\nCollecting hidden representations...")

    H_train, y_train = collect_hidden_features(
        model=model,
        loader=train_loader,
        device=device
    )

    print("Training hidden feature shape:", H_train.shape)

    print("\nFitting real-valued least-squares classifier...")

    classifier = fit_least_squares_classifier(
        H=H_train,
        labels=y_train,
        num_classes=10
    )

    print("Least-squares classifier shape:", classifier.shape)

    train_ls_accuracy = evaluate_training_fit(
        H=H_train,
        labels=y_train,
        classifier=classifier
    )

    print(
        f"\nLeast-Squares Training Accuracy: "
        f"{train_ls_accuracy:.2f}%"
    )

    print("\nEvaluating least-squares classifier...")

    test_ls_accuracy = evaluate_least_squares(
        model=model,
        classifier=classifier,
        test_loader=test_loader,
        device=device
    )

    print(
        f"Least-Squares Test Accuracy: "
        f"{test_ls_accuracy:.2f}%"
    )

    difference = test_ls_accuracy - dfa_test_accuracy

    print("\n==========================================")
    print(f"BNN + DFA: {dfa_test_accuracy:.2f}%")
    print(
        f"BNN + DFA + Real-Valued LS Head: "
        f"{test_ls_accuracy:.2f}%"
    )
    print(f"Difference: {difference:+.2f} percentage points")
    print("==========================================")

    append_result(
        experiment="dfa_least_squares",
        method="BNN + DFA + Real-Valued LS Head",
        seed=seed,
        test_accuracy=test_ls_accuracy,
        test_loss=None,
        training_time_sec=training_timer.seconds,
        parameter_count=count_parameters(model),
    )
