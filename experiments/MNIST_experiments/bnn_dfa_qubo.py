import torch
import torch.nn.functional as F

from models.bnn import BNN
from models.dfa import DFAClassifier, DFAFunction
from models.qubo_head import (
    calculate_ls_scales,
    optimize_binary_classifier
)
from utils.MNIST.data import get_mnist_loaders
from utils.MNIST.seed import set_seed, get_seed

from experiments.MNIST_experiments.results_utils import (
    append_result,
    Timer,
    count_parameters,
)


# ============================================================
# TRAIN BNN + DFA
# ============================================================

def train_dfa(
    model,
    train_loader,
    optimizer,
    dfa,
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

        hidden_pre = model.fc1(x)

        hidden = torch.relu(
            hidden_pre
        )

        outputs = model.fc2(
            hidden
        )

        loss = F.cross_entropy(
            outputs,
            labels
        )

        # ----------------------------------------------------
        # DFA error
        # ----------------------------------------------------

        output_error = (
            DFAFunction.output_error(
                outputs,
                labels
            )
        )

        hidden_error = (
            dfa.hidden_gradient(
                output_error
            )
        )

        hidden_grad = (
            hidden_error
            * (hidden_pre > 0).float()
        )

        # ----------------------------------------------------
        # Manual DFA gradients
        # ----------------------------------------------------

        grad_w1 = (
            hidden_grad.t() @ x
        )

        grad_b1 = (
            hidden_grad.sum(
                dim=0
            )
        )

        grad_w2 = (
            output_error.t() @ hidden
        )

        grad_b2 = (
            output_error.sum(
                dim=0
            )
        )

        model.fc1.weight.grad = grad_w1
        model.fc1.bias.grad = grad_b1

        model.fc2.weight.grad = grad_w2
        model.fc2.bias.grad = grad_b2

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / len(train_loader),
        100.0 * correct / total
    )


# ============================================================
# COLLECT HIDDEN FEATURES
# ============================================================

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


# ============================================================
# BINARY TARGETS
# ============================================================

def create_binary_targets(
    labels,
    num_classes=10
):

    one_hot = F.one_hot(
        labels,
        num_classes=num_classes
    ).float()

    return (
        2.0 * one_hot
        - 1.0
    )


# ============================================================
# FIT SCALE + BIAS AFTER QUBO
# ============================================================

def calculate_final_scale_and_bias(
    H,
    labels,
    binary_weights,
    num_classes=10
):
    """
    Fit the final continuous scale and bias while keeping
    the QUBO binary weights fixed.

    Target:

        +1 for correct class
        -1 otherwise
    """

    H = H.float()

    W = binary_weights.clone().float()

    targets = create_binary_targets(
        labels,
        num_classes
    )

    scales = torch.zeros(
        num_classes
    )

    biases = torch.zeros(
        num_classes
    )

    for class_id in range(num_classes):

        z = (
            H
            @ W[:, class_id]
        )

        target = (
            targets[:, class_id]
        )

        z_mean = z.mean()
        target_mean = target.mean()

        z_centered = (
            z - z_mean
        )

        target_centered = (
            target - target_mean
        )

        denominator = (
            z_centered
            @ z_centered
        )

        if denominator.item() > 1e-12:

            scale = (
                z_centered
                @ target_centered
            ) / denominator

        else:

            scale = torch.tensor(
                1.0
            )

        bias = (
            target_mean
            - scale * z_mean
        )

        scales[class_id] = scale
        biases[class_id] = bias

    return (
        W,
        scales,
        biases
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_qubo_head(
    model,
    binary_weights,
    scales,
    biases,
    test_loader,
    device
):

    model.eval()

    W = binary_weights.to(
        device
    )

    scales = scales.to(
        device
    )

    biases = biases.to(
        device
    )

    effective_weights = (
        W
        * scales.unsqueeze(0)
    )

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

        logits = (
            hidden
            @ effective_weights
            + biases
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        100.0
        * correct
        / total
    )


# ============================================================
# MAIN
# ============================================================


    append_result(
        experiment="bnn_dfa_qubo",
        method="BNN + DFA + QUBO",
        seed=seed,
        test_accuracy=test_accuracy,
        test_loss=test_loss if "test_loss" in locals() else None,
        training_time_sec=training_timer.seconds if "training_timer" in locals() else None,
        parameter_count=count_parameters(model),
        qubo_time_sec=qubo_timer.seconds if "qubo_timer" in locals() else None,
    )


if __name__ == "__main__":

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    seed = get_seed()
    set_seed(seed)

    print(f"Using seed: {seed}")

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
    # Data
    # --------------------------------------------------------

    train_loader, test_loader = (
        get_mnist_loaders(
            batch_size=64
        )
    )

    # ========================================================
    # STEP 1
    # BNN + DFA
    # ========================================================

    print()
    print(
        "Training BNN + DFA..."
    )

    model = BNN().to(
        device
    )

    dfa = DFAClassifier(
        hidden_size=128,
        output_size=10,
        device=device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    with Timer() as training_timer:
        for epoch in range(16):

            loss, accuracy = train_dfa(
                model,
                train_loader,
                optimizer,
                dfa,
                device
            )

            print(
                f"Epoch {epoch + 1}/16 | "
                f"Train Loss: {loss:.4f} | "
                f"Train Accuracy: "
                f"{accuracy:.2f}%"
            )

    # ========================================================
    # STEP 2
    # HIDDEN REPRESENTATIONS
    # ========================================================

    print()
    print(
        "Collecting hidden representations..."
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

    # ========================================================
    # STEP 3
    # TARGETS
    # ========================================================

    print()
    print(
        "Constructing binary classification targets..."
    )

    targets = create_binary_targets(
        y_train,
        num_classes=10
    )

    print(
        "Target shape:",
        targets.shape
    )

    print(
        "Target values:",
        torch.unique(targets)
    )

    # ========================================================
    # STEP 4
    # CALCULATE NATURAL QUBO SCALES
    # ========================================================

    print()
    print(
        "Calculating QUBO scales from "
        "real-valued least-squares solution..."
    )

    qubo_scales = calculate_ls_scales(
        H=H_train,
        labels=y_train,
        num_classes=10
    )

    print()
    print(
        "QUBO scales:"
    )

    for class_id in range(10):

        print(
            f"  Class {class_id}: "
            f"{qubo_scales[class_id].item():.6f}"
        )

    # ========================================================
    # STEP 5
    # QUBO
    # ========================================================

    print()
    print(
        "Optimizing binary classifier using QUBO..."
    )

    with Timer() as qubo_timer:
        binary_weights = optimize_binary_classifier(
            H=H_train,
            targets=targets,
            scales=qubo_scales,
            num_classes=10,
            num_reads=100,
            seed=seed
        )

    # ========================================================
    # VERIFY BINARY WEIGHTS
    # ========================================================

    print()
    print(
        "Binary weight shape:",
        binary_weights.shape
    )

    print(
        "Unique binary weight values:",
        torch.unique(
            binary_weights
        )
    )

    # ========================================================
    # STEP 6
    # FINAL SCALE + BIAS
    # ========================================================

    print()
    print(
        "Optimizing final class-specific "
        "scales and biases..."
    )

    (
        binary_weights,
        final_scales,
        biases
    ) = calculate_final_scale_and_bias(
        H=H_train,
        labels=y_train,
        binary_weights=binary_weights,
        num_classes=10
    )

    print()
    print(
        "Final scales:"
    )

    for class_id in range(10):

        print(
            f"  Class {class_id}: "
            f"{final_scales[class_id].item():.6f}"
        )

    print()
    print(
        "Final biases:"
    )

    for class_id in range(10):

        print(
            f"  Class {class_id}: "
            f"{biases[class_id].item():.6f}"
        )

    # ========================================================
    # STEP 7
    # EVALUATE
    # ========================================================

    print()
    print(
        "Evaluating QUBO binary classifier..."
    )

    test_accuracy = evaluate_qubo_head(
        model=model,
        binary_weights=binary_weights,
        scales=final_scales,
        biases=biases,
        test_loader=test_loader,
        device=device
    )

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "BNN + DFA + QUBO Binary Head "
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        "=========================================="
    )

    append_result(
        experiment="bnn_dfa_qubo",
        method="BNN + DFA + QUBO",
        seed=seed,
        test_accuracy=test_accuracy,
        test_loss=None,
        training_time_sec=training_timer.seconds,
        parameter_count=count_parameters(model),
        qubo_time_sec=qubo_timer.seconds,
    )
