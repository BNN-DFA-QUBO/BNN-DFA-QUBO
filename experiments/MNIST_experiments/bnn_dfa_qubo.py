import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

from models.bnn import BNN
from models.dfa import DFAClassifier, DFAFunction
from models.qubo_head import (
    calculate_ridge_ls_scales,
    calculate_legacy_ls_scales,
    optimize_binary_classifier,
)
from utils.MNIST.data import get_mnist_loaders
from utils.MNIST.seed import set_seed

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


def _recorded_qubo_runtime(tuning_rows, fixed_runtime_sec, total_tuning_seconds=None):
    """Return end-to-end tuner duration, or the fixed-run solver timer.

    Tuned duration includes scale initialization, all candidate fits/refits,
    validation scoring, and winner selection. Test evaluation is excluded.
    """
    if tuning_rows is None:
        return fixed_runtime_sec
    if total_tuning_seconds is not None:
        return float(total_tuning_seconds)
    return sum(float(row["runtime_sec"]) for row in tuning_rows)


def _append_qubo_result(tuning_rows, fixed_runtime_sec, total_tuning_seconds=None, **result_fields):
    """Append a QUBO result with runtime semantics matching its run mode."""
    result_fields["qubo_time_sec"] = _recorded_qubo_runtime(
        tuning_rows, fixed_runtime_sec, total_tuning_seconds,
    )
    if tuning_rows is not None and result_fields.get("qubo_selected_candidate_time_sec") is None:
        result_fields["qubo_selected_candidate_time_sec"] = next(
            (float(row["runtime_sec"]) for row in tuning_rows if row.get("selected")), None
        )
    append_result(**result_fields)


if __name__ == "__main__":

    # Import after this module's training and QUBO helpers are defined. The
    # shared tuner imports those helpers for its standalone diagnostic path.
    from experiments.MNIST_experiments.tune_qubo_validation import (
        DEFAULT_NUM_READS,
        DEFAULT_NUM_SWEEPS,
        DEFAULT_SCALE_MULTIPLIERS,
        _parse_grid,
        tune_from_representations,
        write_tuning_csv,
        validate_grid,
    )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(description="MNIST BNN+DFA QUBO head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qubo-scale-solver", choices=("ridge", "legacy"), default="ridge",
                        help="LS solver used only to initialize QUBO alpha scales")
    parser.add_argument("--qubo-tuning", action="store_true",
                        help="select QUBO reads and scale multiplier using a stratified training-only validation split")
    parser.add_argument("--num-reads", type=int, default=100,
                        help="fixed QUBO reads when --qubo-tuning is disabled")
    parser.add_argument("--num-sweeps", type=int, default=1000,
                        help="fixed QUBO sweeps when --qubo-tuning is disabled")
    parser.add_argument("--num-reads-grid", type=lambda s: _parse_grid(s, int, "num-reads-grid"),
                        default=DEFAULT_NUM_READS, help="tuning grid: comma-separated positive integers")
    parser.add_argument("--num-sweeps-grid", type=lambda s: _parse_grid(s, int, "num-sweeps-grid"),
                        default=DEFAULT_NUM_SWEEPS, help="tuning grid: comma-separated positive integers")
    parser.add_argument("--scale-multipliers", type=lambda s: _parse_grid(s, float, "scale-multipliers"),
                        default=DEFAULT_SCALE_MULTIPLIERS, help="tuning grid: comma-separated positive numbers")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--representation-dir", type=Path, default=Path("data/mnist_representations"))
    args = parser.parse_args()
    if args.epochs < 1 or args.num_reads < 1 or args.num_sweeps < 1:
        parser.error("epochs, num-reads, and num-sweeps must be positive")
    if not 0.05 <= args.validation_fraction <= 0.4:
        parser.error("validation-fraction must be between 0.05 and 0.4")
    if args.qubo_tuning:
        try:
            validate_grid(args.num_reads_grid, args.num_sweeps_grid, args.scale_multipliers)
        except ValueError as exc:
            parser.error(str(exc))
        if args.qubo_scale_solver != "ridge":
            parser.error("--qubo-tuning requires the Ridge scale initializer")
    seed = args.seed
    scale_solver = args.qubo_scale_solver
    set_seed(seed)

    print(f"Using seed: {seed}")
    print(f"QUBO scale initializer: {scale_solver}")

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

    base_train_loader, test_loader = (
        get_mnist_loaders(
            batch_size=64
        )
    )
    dataset = base_train_loader.dataset
    training_seconds = 0.0
    representation = None
    representation_path = ""
    representation_source = ""

    # ========================================================
    # STEP 1
    # BNN + DFA
    # ========================================================

    print()
    print(
        "Training BNN + DFA..."
    )

    if args.qubo_tuning:
        from experiments.MNIST_experiments.mnist_representation import load_or_train_mnist_representation
        representation, representation_path, representation_source = load_or_train_mnist_representation(
            dataset, test_loader.dataset, seed, args.epochs, args.validation_fraction,
            device=device, output_dir=args.representation_dir,
        )
        H_train, y_train = representation["H_train"], representation["y_train"]
        H_val, y_val = representation["H_validation"], representation["y_validation"]
        H_test, y_test = representation["H_test"], representation["y_test"]
        training_seconds = (float(representation.get("training_seconds", 0.0))
                            if representation_source == "trained" else 0.0)
        model = None
        print(f"Representation: {representation_path.resolve()} ({representation_source}, "
              f"id={representation['artifact_id']})")
        print(f"Tuning protocol: stratified train/validation split; "
              f"{1.0 - args.validation_fraction:.0%} official training examples used for fitting")
    else:
        # Preserve the historical fixed-configuration protocol: train on all official train data.
        train_loader = base_train_loader
        model = BNN().to(device)
        dfa = DFAClassifier(hidden_size=128, output_size=10, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with Timer() as training_timer:
            for epoch in range(args.epochs):
                loss, accuracy = train_dfa(model, train_loader, optimizer, dfa, device)
                print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {loss:.4f} | Train Accuracy: {accuracy:.2f}%")
        training_seconds = training_timer.seconds

    # ========================================================
    # STEP 2
    # HIDDEN REPRESENTATIONS
    # ========================================================

    print()
    print(
        "Collecting hidden representations..."
    )

    if not args.qubo_tuning:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with Timer() as extraction_timer:
            H_train, y_train = collect_hidden_features(model, train_loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        extraction_seconds = extraction_timer.seconds
    else:
        extraction_seconds = (float(representation.get("feature_extraction_seconds", 0.0))
                              if representation_source == "trained" else 0.0)

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
        f"Calculating QUBO scales using {scale_solver} LS initialization..."
    )

    tuning_winner = None
    tuning_rows = None
    if args.qubo_tuning:
        tuning_winner, tuning_rows = tune_from_representations(
            H_train, y_train, H_val, y_val, seed,
            args.num_reads_grid, args.num_sweeps_grid, args.scale_multipliers,
        )
        qubo_scales = None
        result_method = "BNN + DFA + QUBO (validation-selected Ridge scale initialization)"
    elif scale_solver == "ridge":
        qubo_scales = calculate_ridge_ls_scales(
            H=H_train, labels=y_train, num_classes=10
        )
        result_method = "BNN + DFA + QUBO (Ridge LS scale initialization)"
    else:
        qubo_scales = calculate_legacy_ls_scales(
            H=H_train, labels=y_train, num_classes=10
        )
        result_method = "BNN + DFA + QUBO (legacy LS scale initialization)"

    print()
    print(
        "QUBO scales:"
    )

    for class_id in range(10) if qubo_scales is not None else []:

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

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with Timer() as qubo_timer:
        if args.qubo_tuning:
            binary_weights = tuning_winner["_weights"]
            final_scales = tuning_winner["_scales"]
            biases = tuning_winner["_bias"]
        else:
            binary_weights = optimize_binary_classifier(
                H=H_train,
                targets=targets,
                scales=qubo_scales,
                num_classes=10,
                num_reads=args.num_reads,
                num_sweeps=args.num_sweeps,
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

    with Timer() as final_scale_bias_timer:
        if not args.qubo_tuning:
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)

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
        model=model, binary_weights=binary_weights, scales=final_scales,
        biases=biases, test_loader=test_loader, device=device
    ) if not args.qubo_tuning else float(
        ( (H_test.float() @ (binary_weights * final_scales.unsqueeze(0)) + biases).argmax(1) == y_test )
        .float().mean().item() * 100.0
    )

    tuning_csv_path = ""
    if args.qubo_tuning:
        tuning_csv_path = write_tuning_csv(
            tuning_rows, seed, args.epochs, args.validation_fraction,
            representation_artifact_id=representation["artifact_id"],
            n_train=representation["metadata"]["sample_counts"]["train"],
            n_validation=representation["metadata"]["sample_counts"]["validation"],
            n_test=representation["metadata"]["sample_counts"]["test"],
        )
        print(f"Tuning configurations CSV: {tuning_csv_path.resolve()}")

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

    backbone_params = (count_parameters(model) if model is not None else
                       int(representation["metadata"]["backbone_parameter_count"]))
    # 128*10 binary weights + 10 scales + 10 biases = 1300
    head_params = int(binary_weights.numel() + final_scales.numel() + biases.numel())
    total_params = backbone_params + head_params

    # Head fit time:
    if args.qubo_tuning:
        head_fit_time = (float(tuning_winner.get("tuning_total_seconds"))
                         if (tuning_winner and tuning_winner.get("tuning_total_seconds") is not None)
                         else sum(float(r["runtime_sec"]) for r in tuning_rows))
    else:
        head_fit_time = qubo_timer.seconds + final_scale_bias_timer.seconds

    # Current-run work
    cur_backbone_time = training_seconds
    cur_extraction_time = extraction_seconds
    cur_total_time = cur_backbone_time + cur_extraction_time + head_fit_time

    # Pipeline timing completeness
    if not args.qubo_tuning or representation_source == "trained":
        e2e_time = cur_total_time
        timing_status = "complete"
    else:
        # Reused representation
        raw_train = representation.get("training_seconds")
        raw_ext = representation.get("feature_extraction_seconds")
        if raw_train is not None and raw_ext is not None:
            e2e_time = float(raw_train) + float(raw_ext) + head_fit_time
            timing_status = "complete"
        elif raw_train is not None:
            e2e_time = None
            timing_status = "missing_extraction_time"
        else:
            e2e_time = None
            timing_status = "missing_backbone_and_extraction_time"

    _append_qubo_result(
        tuning_rows=tuning_rows,
        fixed_runtime_sec=qubo_timer.seconds,
        total_tuning_seconds=(tuning_winner.get("tuning_total_seconds") if tuning_winner else None),
        experiment="bnn_dfa_qubo",
        method=result_method,
        seed=seed,
        test_accuracy=test_accuracy,
        test_loss=None,
        training_time_sec=training_seconds,
        parameter_count=backbone_params,
        backbone_training_time_sec=cur_backbone_time,
        feature_extraction_time_sec=cur_extraction_time,
        head_fit_time_sec=head_fit_time,
        current_run_total_time_sec=cur_total_time,
        end_to_end_training_time_sec=e2e_time,
        pipeline_timing_status=timing_status,
        backbone_parameter_count=backbone_params,
        head_parameter_count=head_params,
        total_parameter_count=total_params,
        qubo_tuning_enabled=args.qubo_tuning,
        qubo_num_reads_selected=tuning_winner["num_reads"] if tuning_winner else None,
        qubo_num_sweeps_selected=tuning_winner["num_sweeps"] if tuning_winner else None,
        qubo_grid_protocol="fixed_27_configuration_v1" if tuning_winner else "",
        qubo_scale_multiplier_selected=tuning_winner["scale_multiplier"] if tuning_winner else None,
        qubo_validation_accuracy=tuning_winner["validation_accuracy"] if tuning_winner else None,
        qubo_validation_fraction=args.validation_fraction if args.qubo_tuning else None,
        qubo_grid_size=len(tuning_rows) if tuning_rows else None,
        qubo_solver_seed_selected=tuning_winner["solver_seed"] if tuning_winner else None,
        qubo_selection_metric="validation_accuracy_percent" if tuning_winner else "",
        notes=f"tuning_csv={tuning_csv_path.resolve()}" if tuning_csv_path else "",
        representation_artifact_id=representation["artifact_id"] if representation else "",
        representation_path=str(representation_path.resolve()) if representation else "",
        representation_cache_status=representation_source,
        n_train=representation["metadata"]["sample_counts"]["train"] if representation else len(y_train),
        n_validation=representation["metadata"]["sample_counts"]["validation"] if representation else None,
        n_test=representation["metadata"]["sample_counts"]["test"] if representation else len(test_loader.dataset),
    )
