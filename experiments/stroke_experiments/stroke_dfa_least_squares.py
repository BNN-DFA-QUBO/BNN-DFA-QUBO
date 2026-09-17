import os

import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


representation_path = (
    "data/stroke_dfa_representation.pt"
)


data = torch.load(
    representation_path,
    weights_only=False
)


H_train = data["H_train"].float()
H_val = data["H_val"].float()
H_test = data["H_test"].float()

y_train = data["y_train"].float()
y_val = data["y_val"].float()
y_test = data["y_test"].float()


print("Representation loaded from:")
print(representation_path)

print("\nRepresentation Shapes")

print(
    "H_train:",
    H_train.shape
)

print(
    "H_val:",
    H_val.shape
)

print(
    "H_test:",
    H_test.shape
)


H_train_augmented = torch.cat(
    [
        H_train,
        torch.ones(
            H_train.size(0),
            1
        )
    ],
    dim=1
)

H_val_augmented = torch.cat(
    [
        H_val,
        torch.ones(
            H_val.size(0),
            1
        )
    ],
    dim=1
)

H_test_augmented = torch.cat(
    [
        H_test,
        torch.ones(
            H_test.size(0),
            1
        )
    ],
    dim=1
)


y_train_signed = (
    2 * y_train - 1
).unsqueeze(1)


A = (
    H_train_augmented.T
    @ H_train_augmented
)

B = (
    H_train_augmented.T
    @ y_train_signed
)


weights = torch.linalg.solve(
    A,
    B
)


train_scores = (
    H_train_augmented @ weights
).squeeze(1)

val_scores = (
    H_val_augmented @ weights
).squeeze(1)

test_scores = (
    H_test_augmented @ weights
).squeeze(1)


thresholds = torch.unique(
    val_scores
).sort().values


best_threshold = 0.0
best_val_balanced_accuracy = -1.0


for threshold in thresholds:

    val_predictions = (
        val_scores >= threshold
    ).float()

    val_balanced_accuracy = (
        balanced_accuracy_score(
            y_val.numpy(),
            val_predictions.numpy()
        )
    )

    if (
        val_balanced_accuracy
        > best_val_balanced_accuracy
    ):

        best_val_balanced_accuracy = (
            val_balanced_accuracy
        )

        best_threshold = (
            threshold.item()
        )


train_predictions = (
    train_scores >= best_threshold
).float()

val_predictions = (
    val_scores >= best_threshold
).float()

test_predictions = (
    test_scores >= best_threshold
).float()


def calculate_metrics(
    labels,
    predictions,
    scores
):

    labels = labels.numpy()
    predictions = predictions.numpy()
    scores = scores.numpy()

    return {
        "balanced_accuracy": balanced_accuracy_score(
            labels,
            predictions
        ),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            labels,
            scores
        ),
        "pr_auc": average_precision_score(
            labels,
            scores
        )
    }


train_metrics = calculate_metrics(
    y_train,
    train_predictions,
    train_scores
)

val_metrics = calculate_metrics(
    y_val,
    val_predictions,
    val_scores
)

test_metrics = calculate_metrics(
    y_test,
    test_predictions,
    test_scores
)


print("\nThreshold Selection")

print(
    f"Selected validation threshold: "
    f"{best_threshold:.6f}"
)

print(
    f"Validation Balanced Accuracy at "
    f"selected threshold: "
    f"{best_val_balanced_accuracy * 100:.2f}%"
)


print("\nReal LS Results")


print("\nTrain Results")

print(
    f"Balanced Accuracy: "
    f"{train_metrics['balanced_accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{train_metrics['precision'] * 100:.2f}%"
)

print(
    f"Recall: "
    f"{train_metrics['recall'] * 100:.2f}%"
)

print(
    f"F1: "
    f"{train_metrics['f1'] * 100:.2f}%"
)

print(
    f"ROC-AUC: "
    f"{train_metrics['roc_auc'] * 100:.2f}%"
)

print(
    f"PR-AUC: "
    f"{train_metrics['pr_auc'] * 100:.2f}%"
)


print("\nValidation Results")

print(
    f"Balanced Accuracy: "
    f"{val_metrics['balanced_accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{val_metrics['precision'] * 100:.2f}%"
)

print(
    f"Recall: "
    f"{val_metrics['recall'] * 100:.2f}%"
)

print(
    f"F1: "
    f"{val_metrics['f1'] * 100:.2f}%"
)

print(
    f"ROC-AUC: "
    f"{val_metrics['roc_auc'] * 100:.2f}%"
)

print(
    f"PR-AUC: "
    f"{val_metrics['pr_auc'] * 100:.2f}%"
)


print("\nTest Results")

print(
    f"Balanced Accuracy: "
    f"{test_metrics['balanced_accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{test_metrics['precision'] * 100:.2f}%"
)

print(
    f"Recall: "
    f"{test_metrics['recall'] * 100:.2f}%"
)

print(
    f"F1: "
    f"{test_metrics['f1'] * 100:.2f}%"
)

print(
    f"ROC-AUC: "
    f"{test_metrics['roc_auc'] * 100:.2f}%"
)

print(
    f"PR-AUC: "
    f"{test_metrics['pr_auc'] * 100:.2f}%"
)


print("\nPrediction Distribution")

test_scores_np = (
    test_scores.numpy()
)

test_predictions_np = (
    test_predictions.numpy()
)

y_test_np = (
    y_test.numpy()
)

print(
    "Minimum score:",
    test_scores_np.min()
)

print(
    "Maximum score:",
    test_scores_np.max()
)

print(
    "Mean score:",
    test_scores_np.mean()
)

print(
    "Predicted positives:",
    test_predictions_np.sum()
)

print(
    "Actual positives:",
    y_test_np.sum()
)


os.makedirs(
    "data",
    exist_ok=True
)


results_path = (
    "data/stroke_dfa_real_ls_results.pt"
)


torch.save(
    {
        "head": "real_ls",

        "weights": weights,

        "threshold": best_threshold,

        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    },
    results_path
)


print("\nResults saved to:")
print(results_path)