import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

import random
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
    average_precision_score
)

from utils.stroke_data import prepare_stroke_data


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


def load_data():
    representation = torch.load(
        "data/stroke_dfa_representation.pt",
        map_location="cpu"
    )

    H_train = representation["H_train"].float()
    H_val = representation["H_val"].float()
    H_test = representation["H_test"].float()

    (
        _,
        _,
        _,
        y_train,
        y_val,
        y_test
    ) = prepare_stroke_data()

    y_train = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32
    )

    y_val = torch.tensor(
        y_val.to_numpy(),
        dtype=torch.long
    )

    y_test = torch.tensor(
        y_test.to_numpy(),
        dtype=torch.long
    )

    return (
        H_train,
        H_val,
        H_test,
        y_train,
        y_val,
        y_test
    )


def find_best_threshold(y_true, scores):
    thresholds = torch.unique(scores)

    best_threshold = thresholds[0].item()
    best_ba = -1.0

    for threshold in thresholds:

        predictions = (
            scores >= threshold
        ).long()

        ba = balanced_accuracy_score(
            y_true.numpy(),
            predictions.numpy()
        )

        if ba > best_ba:
            best_ba = ba
            best_threshold = threshold.item()

    return best_threshold


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
            average_precision_score(y, s)
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

    positive_weight = (
        (labels == 0).sum()
        / (labels == 1).sum()
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
                target,
                pos_weight=positive_weight
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

    positive_weight = (
        (labels == 0).sum()
        / (labels == 1).sum()
    )

    head = STEBinaryHead(
        H.shape[1]
    )

    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=0.01
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    )

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

def main():

    (
        H_train,
        H_val,
        H_test,
        y_train,
        y_val,
        y_test
    ) = load_data()

    print("H_train:", H_train.shape)
    print("H_val:", H_val.shape)
    print("H_test:", H_test.shape)

    results = []

    for seed in SEEDS:

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

    df.to_csv(
        "data/stroke_binary_head_multiseed.csv",
        index=False
    )

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

    summary.to_csv(
        "data/stroke_binary_head_multiseed_summary.csv"
    )

    print()
    print(
        "Saved:"
    )

    print(
        "data/stroke_binary_head_multiseed.csv"
    )

    print(
        "data/stroke_binary_head_multiseed_summary.csv"
    )


if __name__ == "__main__":
    main()