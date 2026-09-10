from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def _to_numpy(array):
    """
    Convert input to a numpy array.
    Works with numpy arrays and pytorch tensors.
    """

    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()

    return np.asarray(array)


def _apply_mask(values, mask):
    """
    Keep only samples with valid targets (where mask == 1).

    mask:
        [N] boolean / 0-1 array
    """
    values = _to_numpy(values)

    if mask is None:
        return values

    mask = _to_numpy(mask).astype(bool)
    return values[mask]


# GENRE - 19 label Multi-class classification


def calculate_genre_probabilities(
    probabilities,
    targets,
    mask=None,
):
    """ "
    Calculate metrics for 19-label multi-label genre classificiation.

    Probabilities:
        [N,19] sigmoid probabilities for each independent 19 labels.

    Targets:
        [N,19] binary ground-truth labels.

    Mask:
        [N] validity mask.
        Only samples with mask value == 1 are evaulated.
    """

    probabilities = _to_numpy(probabilities)
    targets = _to_numpy(targets).astype[int]

    if probabilities.ndim != 2 or probabilities.shape[1] != 19:
        raise ValueError(
            f"Genre probabilities must have shape [N,19]. Given : {probabilities.shape}"
        )

    if targets.ndim != 2 or targets.shape[1] != 19:
        raise ValueError(f"Genre targets must have shape [N,19],Given : {targets.shape}")

    # Apply sample-level mask before evaluation.
    if mask is not None:
        probabilities = _apply_mask(probabilities, mask)
        targets = _apply_mask(targets, mask)

    # Phase 1 uses a fixed threshold of 0.5
    # Will be configured later in phase 2
    predictions = (probabilities >= 0.5).astype(int)

    # Overall metrics
    metrics: dict[str, Any] = {
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                targets,
                predictions,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }

    # Per-Genre metrics
    f1 = f1_score(
        targets,
        predictions,
        average=None,
        zero_division=0,
    )

    precision = precision_score(
        targets,
        predictions,
        average=None,
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        average=None,
        zero_division=0,
    )

    metrics["per_genre"] = {}

    for i in range(19):
        metrics["per_genre"][f"genre_{i}"] = {
            "f1": float(f1[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
        }

    return metrics


# Normal multi class classification


def calculate_classification_metrics(
    probabilities,
    targets,
    mask=None,
    class_names=None,
):
    """
    Calculate metrics for a multi class classification task.

    probabilities:
        [N,C] softmax probabilities.

    targets:
        [N] integer class labels.

    mask:
        [N] validity mask.

    """

    probabilities = _to_numpy(probabilities)
    targets = _to_numpy(targets).astype(int)

    # Check Shapes
    if probabilities.ndim != 2:
        raise ValueError(f"Probabilities must have shape [N,C] got {probabilities.shape}")

    if targets.ndim != 1:
        raise ValueError(f"Targets must have shape [N], got {targets.shape}")

    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Number of samples must match: {probabilities.shape[0]} vs {targets.shape[0]}"
        )

    # Apply mask
    probabilities = _apply_mask(probabilities, mask)
    targets = _apply_mask(targets, mask)

    # Highest probablity = predicted class
    predictions = np.argmax(probabilities, axis=1)

    num_classes = probabilities.shape[1]

    # Default class names
    if class_names is None:
        class_names = []
        for i in range(num_classes):
            class_names.append(f"class_{i}")

    if len(class_names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(class_names)}")

    # Overall metrics
    metrics: dict[str, Any] = {
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                targets,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                targets,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                targets,
                predictions,
            )
        ),
        "macro_precision": float(
            precision_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }

    # Per class matrices
    report = classification_report(
        targets,
        predictions,
        labels=np.arange(num_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics["per_class"] = {}
    for class_name in class_names:
        metrics["per_class"][f"class_{class_name}"] = {
            "f1": float(report[class_name]["f1-score"]),
            "precision": float(report[class_name]["precision"]),
            "recall": float(report[class_name]["recall"]),
            "support": int(report[class_name]["support"]),
        }

    return metrics


# Box-OFffice : 4-Class Classification


def calculate_box_office_metrics(
    probabilities,
    targets,
    mask,
):
    """
    Metric for 4-class box-office classification.

    Class mapping:
    0 = Flop,
    1 = Average
    2 = Hit
    3 = Blockbuster
    """

    return calculate_classification_metrics(
        probabilities=probabilities,
        targets=targets,
        mask=mask,
        class_names=[
            "Flop",
            "Average",
            "Hit",
            "Blockbuster",
        ],
    )


# Content Rating : 4-Class classification


def calculate_content_rating_metrics(
    probabilities,
    targets,
    mask,
):
    """
    Metrics for 4-class content-rating clssification.

    Class mapping:
    0 = G
    1 = PG
    2 = PG-13
    3 = R
    """

    return calculate_classification_metrics(
        probabilities=probabilities,
        targets=targets,
        mask=mask,
        class_names=[
            "G",
            "PG",
            "PG-13",
            "R",
        ],
    )


# Rating - Regression


def calculate_rating_metrics(predictions, targets, mask):
    """
    Metrics for rating regression.

    Output:
        MAE
        RMSE
    """

    predictions = _to_numpy(predictions).reshape(-1)
    targets = _to_numpy(targets)

    if predictions.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Number of predictions and targets must match"
            f"{predictions.shape[0]} vs {targets.shape[0]}"
        )

    # Apply mask
    predictions = _apply_mask(predictions, mask)
    targets = _apply_mask(targets, mask)

    mse = mean_squared_error(
        targets,
        predictions,
    )

    return {
        "mae": float(
            mean_absolute_error(
                targets,
                predictions,
            )
        ),
        "rmse": float(np.sqrt(mse)),
    }


# Classification Outputs


def get_classification_outputs(
    probabilities,
    targets,
    mask,
    class_names=None,
):
    """
    Return predicted labels, confusion matrix,
    and classification report.
    """

    probabilities = _to_numpy(probabilities)
    targets = _to_numpy(targets).astype(int)

    if probabilities.ndim != 2:
        raise ValueError(f"Probabilities must have shape [N,C], got {probabilities.shape}")

    if targets.ndim != 1:
        raise ValueError(f"Targets must have shape [N], got {targets.shape}")

    # Apply mask
    probabilities = _apply_mask(probabilities, mask)
    targets = _apply_mask(targets, mask)

    predictions = np.argmax(probabilities, axis=1)

    num_classes = probabilities.shape[1]

    class_names = []
    for i in range(num_classes):
        class_names.append(f"class_{i}")

    if len(class_names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(class_names)}")

    return {
        "predictions": predictions,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=np.arange(num_classes)),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=np.arange(num_classes),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


# Composite or Combined score


def compute_composite_score(
    genre_macro_f1,
    rating_mae,
    box_office_macro_f1,
    content_rating_macro_f1,
    rating_max_error,
):
    """
    Calculates a single score from combining these metrics
    with each config for it's each fold.

    Classification metrics are already in [0,1] :
        Because metrics used of those 3 tasks naturally
        lie between [0,1].

    Rating MAE will be converted into a score lying between
    [0,1] as well.

    Now all four metrics will have same weight for computation.
    """

    if rating_max_error <= 0:
        raise ValueError("rating_mae_baseline must be greater than 0.")

    rating_score = max(
        0.0,
        1.0 - rating_mae / rating_max_error,
    )

    return float(
        (genre_macro_f1 + rating_score + box_office_macro_f1 + content_rating_macro_f1) / 4.0
    )
