import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

import random
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import neal

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from utils.stroke_representation import load_representation
from utils.stroke_data import DATA_DIR
from utils.stroke_metrics import find_best_threshold as shared_find_best_threshold



# These are head initialization/optimization seeds on one fixed representation.
SEEDS = [42, 123, 2024, 7, 99]
NUM_READS = 100
STE_EPOCHS = 200
DIRECT_MAX_PASSES = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(representation_path):
    representation = load_representation(representation_path)

    H_train = representation["H_train"].float()
    H_val = representation["H_val"].float()
    H_test = representation["H_test"].float()

    # Labels are stored with the representations so a separately regenerated split
    # can never silently misalign examples and learned features.
    y_train = representation["y_train"].float()
    y_val = representation["y_val"].long()
    y_test = representation["y_test"].long()
    if not (len(H_train) == len(y_train) and len(H_val) == len(y_val) and len(H_test) == len(y_test)):
        raise ValueError("Representation rows and embedded labels are misaligned")

    return (
        H_train,
        H_val,
        H_test,
        y_train,
        y_val,
        y_test,
        representation
    )


def find_best_threshold(y_true, scores):
    threshold, _ = shared_find_best_threshold(y_true.numpy(), scores.numpy())
    return float(threshold)


def evaluate(
    y_true,
    scores,
    threshold
):
    predictions = (
        scores >= threshold
    ).long()

    y = y_true.numpy()
    p = predictions.numpy()
    s = scores.numpy()
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()

    return {
        "balanced_accuracy":
            balanced_accuracy_score(y, p),

        "precision":
            precision_score(
                y,
                p,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                p,
                zero_division=0
            ),

        "f1":
            f1_score(
                y,
                p,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(y, s),

        "pr_auc":
            average_precision_score(y, s),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold)
    }


# ============================================================
# DIRECT BINARY
# ============================================================

def direct_binary(
    H,
    labels,
    seed
):
    set_seed(seed)

    weights = torch.ones(
        H.shape[1],
        dtype=torch.float32
    )

    target = labels.clone()

    best_loss = float("inf")

    for _ in range(DIRECT_MAX_PASSES):

        improved = False

        for i in torch.randperm(
            H.shape[1]
        ).tolist():

            candidate = weights.clone()

            candidate[i] *= -1

            scores = H @ candidate

            loss = nn.functional.binary_cross_entropy_with_logits(
                scores,
                target
            )

            if loss.item() < best_loss:

                best_loss = loss.item()
                weights = candidate

                improved = True

        if not improved:
            break

    return weights


# ============================================================
# STE BINARY
# ============================================================

class STEBinaryHead(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(input_size) * 0.01
        )

        self.bias = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(self, x):

        binary_weight = self.weight.sign()

        binary_weight = (
            binary_weight.detach()
            - self.weight.detach()
            + self.weight
        )

        return (
            x @ binary_weight
            + self.bias
        )


def ste_binary(
    H,
    labels,
    seed
):
    set_seed(seed)

    head = STEBinaryHead(
        H.shape[1]
    )

    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=0.01
    )

    criterion = nn.BCEWithLogitsLoss()

    for _ in range(STE_EPOCHS):

        optimizer.zero_grad()

        logits = head(H)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

    with torch.no_grad():

        weights = head.weight.sign().detach()
        bias = head.bias.detach().item()

    return weights, bias


# ============================================================
# QUBO
# ============================================================

def calculate_alpha(H, labels):

    H_aug = torch.cat(
        [
            H,
            torch.ones(
                H.shape[0],
                1
            )
        ],
        dim=1
    )

    Y = torch.nn.functional.one_hot(
        labels.long(),
        num_classes=2
    ).float()

    W_ls = torch.linalg.lstsq(
        H_aug,
        Y
    ).solution

    W_positive = W_ls[
        :H.shape[1],
        1
    ]

    return (
        torch.linalg.vector_norm(
            W_positive
        )
        / (H.shape[1] ** 0.5)
    ).item()


def build_ising_model(
    H,
    target,
    alpha
):

    Hc = (
        H
        - H.mean(
            dim=0,
            keepdim=True
        )
    )

    yc = (
        target
        - target.mean()
    )

    gram = Hc.T @ Hc

    quadratic = (
        alpha ** 2
        * gram
    )

    linear = (
        -2.0
        * alpha
        * (Hc.T @ yc)
    )

    hidden_size = H.shape[1]

    h = {}

    J = {}

    for i in range(hidden_size):

        h[i] = float(
            linear[i].item()
        )

    for i in range(hidden_size):

        for j in range(i + 1, hidden_size):

            coefficient = (
                2.0
                * quadratic[i, j]
            ).item()

            if abs(coefficient) > 1e-12:

                J[(i, j)] = float(
                    coefficient
                )

    return h, J


def qubo_binary(
    H,
    labels,
    seed
):
    set_seed(seed)

    target = (
        labels * 2.0
        - 1.0
    )

    alpha = calculate_alpha(
        H,
        labels
    )

    h, J = build_ising_model(
        H,
        target,
        alpha
    )

    sampler = (
        neal.SimulatedAnnealingSampler()
    )

    response = sampler.sample_ising(
        h,
        J,
        num_reads=NUM_READS,
        seed=seed
    )

    sample = response.first.sample

    weights = torch.tensor(
        [
            1.0
            if sample[i] == 1
            else -1.0
            for i in range(H.shape[1])
        ],
        dtype=torch.float32
    )

    bias = (
        target.mean()
        - alpha
        * (
            H.mean(dim=0)
            @ weights
        )
    ).item()

    return weights, alpha, bias


# ============================================================
# MAIN
# ============================================================

def main(representation_path=DATA_DIR / "stroke_dfa_representation_seed_42.pt", seeds=None,
         output_path=DATA_DIR / "stroke_binary_head_multiseed.csv", overwrite=False):
    summary_path = str(Path(output_path).with_name(Path(output_path).stem + "_summary.csv"))
    existing = [path for path in (output_path, summary_path) if Path(path).exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to replace existing multi-seed outputs: " + ", ".join(existing) + "; pass --overwrite")

    (
        H_train,
        H_val,
        H_test,
        y_train,
        y_val,
        y_test,
        representation
    ) = load_data(representation_path)
    seeds = list(SEEDS if seeds is None else seeds)
    provenance = representation["data_metadata"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("H_train:", H_train.shape)
    print("H_val:", H_val.shape)
    print("H_test:", H_test.shape)

    results = []

    for seed in seeds:

        print()
        print("=" * 60)
        print(f"SEED {seed}")
        print("=" * 60)

        # ----------------------------------------------------
        # Direct Binary
        # ----------------------------------------------------

        weights = direct_binary(
            H_train,
            y_train,
            seed
        )

        train_scores = H_train @ weights
        val_scores = H_val @ weights
        test_scores = H_test @ weights

        threshold = find_best_threshold(
            y_val,
            val_scores
        )

        metrics = evaluate(
            y_test,
            test_scores,
            threshold
        )

        results.append(
            {
                "seed": seed,
                "head_seed": seed,
                "model_seed": representation["training_seed"],
                "training_seed": representation["training_seed"],
                "representation_seed": representation["training_seed"],
                "split_seed": representation["split_seed"],
                "dataset_sha256": representation["dataset_sha256"],
                "preprocessing_version": representation["preprocessing_version"],
                "balancing_method": representation["balancing_method"],
                "experiment_type": "fixed_representation_head_seed_variability",
                "representation_path": str(representation_path),
                "original_train_class_counts": json.dumps(provenance["original_train_class_counts"], sort_keys=True),
                "resampled_train_class_counts": json.dumps(provenance["resampled_train_class_counts"], sort_keys=True),
                "validation_class_counts": json.dumps(provenance["validation_class_counts"], sort_keys=True),
                "test_class_counts": json.dumps(provenance["test_class_counts"], sort_keys=True),
                "loss_configuration": "unweighted BCE on balanced hidden representations",
                "method": "Direct Binary",
                **metrics
            }
        )

        print(
            f"Direct Binary: "
            f"BA={metrics['balanced_accuracy'] * 100:.4f}%"
        )

        # ----------------------------------------------------
        # STE Binary
        # ----------------------------------------------------

        weights, bias = ste_binary(
            H_train,
            y_train,
            seed
        )

        train_scores = (
            H_train @ weights
            + bias
        )

        val_scores = (
            H_val @ weights
            + bias
        )

        test_scores = (
            H_test @ weights
            + bias
        )

        threshold = find_best_threshold(
            y_val,
            val_scores
        )

        metrics = evaluate(
            y_test,
            test_scores,
            threshold
        )

        results.append(
            {
                "seed": seed,
                "head_seed": seed,
                "model_seed": representation["training_seed"],
                "training_seed": representation["training_seed"],
                "representation_seed": representation["training_seed"],
                "split_seed": representation["split_seed"],
                "dataset_sha256": representation["dataset_sha256"],
                "preprocessing_version": representation["preprocessing_version"],
                "balancing_method": representation["balancing_method"],
                "experiment_type": "fixed_representation_head_seed_variability",
                "representation_path": str(representation_path),
                "original_train_class_counts": json.dumps(provenance["original_train_class_counts"], sort_keys=True),
                "resampled_train_class_counts": json.dumps(provenance["resampled_train_class_counts"], sort_keys=True),
                "validation_class_counts": json.dumps(provenance["validation_class_counts"], sort_keys=True),
                "test_class_counts": json.dumps(provenance["test_class_counts"], sort_keys=True),
                "loss_configuration": "unweighted BCE on balanced hidden representations",
                "method": "STE Binary",
                **metrics
            }
        )

        print(
            f"STE Binary: "
            f"BA={metrics['balanced_accuracy'] * 100:.4f}%"
        )

        # ----------------------------------------------------
        # QUBO Binary
        # ----------------------------------------------------

        weights, alpha, bias = qubo_binary(
            H_train,
            y_train,
            seed
        )

        train_scores = (
            alpha
            * (H_train @ weights)
            + bias
        )

        val_scores = (
            alpha
            * (H_val @ weights)
            + bias
        )

        test_scores = (
            alpha
            * (H_test @ weights)
            + bias
        )

        threshold = find_best_threshold(
            y_val,
            val_scores
        )

        metrics = evaluate(
            y_test,
            test_scores,
            threshold
        )

        results.append(
            {
                "seed": seed,
                "head_seed": seed,
                "model_seed": representation["training_seed"],
                "training_seed": representation["training_seed"],
                "representation_seed": representation["training_seed"],
                "split_seed": representation["split_seed"],
                "dataset_sha256": representation["dataset_sha256"],
                "preprocessing_version": representation["preprocessing_version"],
                "balancing_method": representation["balancing_method"],
                "experiment_type": "fixed_representation_head_seed_variability",
                "representation_path": str(representation_path),
                "original_train_class_counts": json.dumps(provenance["original_train_class_counts"], sort_keys=True),
                "resampled_train_class_counts": json.dumps(provenance["resampled_train_class_counts"], sort_keys=True),
                "validation_class_counts": json.dumps(provenance["validation_class_counts"], sort_keys=True),
                "test_class_counts": json.dumps(provenance["test_class_counts"], sort_keys=True),
                "loss_configuration": "QUBO squared-error objective on balanced hidden representations",
                "method": "QUBO Binary",
                "alpha": alpha,
                **metrics
            }
        )

        print(
            f"QUBO Binary: "
            f"BA={metrics['balanced_accuracy'] * 100:.4f}%"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    df = pd.DataFrame(results)

    df.to_csv(output_path, index=False)

    print()
    print("=" * 60)
    print("PER-SEED RESULTS")
    print("=" * 60)

    print(
        df.to_string(
            index=False
        )
    )

    summary = (
        df.groupby("method")
        [
            [
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc"
            ]
        ]
        .agg(["mean", "std"])
    )

    print()
    print("=" * 60)
    print("MEAN ± STANDARD DEVIATION")
    print("=" * 60)

    print(summary)

    summary.to_csv(summary_path)

    print()
    print(
        "Saved:"
    )

    print(
        output_path
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure binary-head initialization variability on one fixed DFA representation")
    parser.add_argument("--representation", default=str(DATA_DIR / "stroke_dfa_representation_seed_42.pt"))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output", default=str(DATA_DIR / "stroke_binary_head_multiseed.csv"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.representation, args.seeds, args.output, args.overwrite)
