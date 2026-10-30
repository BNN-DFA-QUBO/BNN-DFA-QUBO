import argparse
import torch
import torch.nn.functional as F

from models.ls_heads import fit_ridge_classifier

from models.bnn import BNN
from models.dfa import DFAClassifier, DFAFunction
from utils.MNIST.data import get_mnist_loaders
from utils.MNIST.seed import set_seed

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

def fit_legacy_least_squares_classifier(
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


def fit_least_squares_classifier(
    H,
    labels,
    num_classes=10,
    lambda_relative=1e-4,
):
    """Fit the new regularized LS head using the stable shared solver.

    Kept under the old function name so existing call sites continue working.
    New experiment outputs must identify this method as Ridge LS. Use
    fit_legacy_least_squares_classifier() for the historical float32 solver.
    """
    return fit_ridge_classifier(
        H=H,
        labels=labels,
        num_classes=num_classes,
        lambda_relative=lambda_relative,
    )


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

    # The stable head is fitted in CPU float64; inference uses model dtype.
    classifier = classifier.to(device=device, dtype=torch.float32)

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
        H_aug.to(dtype=classifier.dtype) @ classifier
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

    parser = argparse.ArgumentParser(description="MNIST BNN+DFA with LS/Ridge-LS head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ls-solver", choices=("ridge", "legacy"), default="ridge",
                        help="ridge: stable float64 Ridge LS; legacy: original float32 default LS")
    args = parser.parse_args()
    seed = args.seed
    ls_solver = args.ls_solver
    set_seed(seed)

    print(f"Using seed: {seed}")
    print(f"LS solver: {ls_solver}")

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

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with Timer() as extraction_timer:
        H_train, y_train = collect_hidden_features(
            model=model,
            loader=train_loader,
            device=device
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print("Training hidden feature shape:", H_train.shape)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with Timer() as head_fit_timer:
        if ls_solver == "ridge":
            print("\nFitting numerically stable Ridge LS classifier...")
            classifier = fit_least_squares_classifier(
                H=H_train, labels=y_train, num_classes=10
            )
            head_name = "Ridge LS"
            method_name = "BNN + DFA + Ridge LS Head"
            summary_head_name = "BNN + DFA + Ridge LS Head"
        else:
            print("\nFitting legacy float32 least-squares classifier...")
            classifier = fit_legacy_least_squares_classifier(
                H=H_train, labels=y_train, num_classes=10
            )
            head_name = "Legacy LS"
            method_name = "BNN + DFA + Real-Valued LS Head (legacy)"
            summary_head_name = "BNN + DFA + Real-Valued LS Head (legacy)"
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print("Least-squares classifier shape:", classifier.shape)

    train_ls_accuracy = evaluate_training_fit(
        H=H_train,
        labels=y_train,
        classifier=classifier
    )

    print(
        f"\n{head_name} Training Accuracy: "
        f"{train_ls_accuracy:.2f}%"
    )

    print("\nEvaluating least-squares classifier...")

    test_ls_accuracy = evaluate_least_squares(
        model=model,
        classifier=classifier,
        test_loader=test_loader,
        device=device
    )

    accuracy_label = "Ridge LS Test Accuracy" if ls_solver == "ridge" else "Least-Squares Test Accuracy"
    print(f"{accuracy_label}: {test_ls_accuracy:.2f}%")

    difference = test_ls_accuracy - dfa_test_accuracy

    print("\n==========================================")
    print(f"BNN + DFA: {dfa_test_accuracy:.2f}%")
    print(
        f"{summary_head_name}: "
        f"{test_ls_accuracy:.2f}%"
    )
    print(f"Difference: {difference:+.2f} percentage points")
    print("==========================================")

    backbone_params = count_parameters(model)
    head_params = int(classifier.numel())
    total_params = backbone_params + head_params
    total_time = training_timer.seconds + extraction_timer.seconds + head_fit_timer.seconds

    append_result(
        experiment="dfa_least_squares",
        method=method_name,
        seed=seed,
        test_accuracy=test_ls_accuracy,
        test_loss=None,
        training_time_sec=training_timer.seconds,
        parameter_count=backbone_params,
        backbone_training_time_sec=training_timer.seconds,
        feature_extraction_time_sec=extraction_timer.seconds,
        head_fit_time_sec=head_fit_timer.seconds,
        current_run_total_time_sec=total_time,
        end_to_end_training_time_sec=total_time,
        pipeline_timing_status="complete",
        backbone_parameter_count=backbone_params,
        head_parameter_count=head_params,
        total_parameter_count=total_params,
    )
