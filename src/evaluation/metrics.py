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
    roc_auc_score,
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


# Safe One-vs-Rest ROC-AUC


def _safe_ovr_roc_auc_values(
    binary_targets,
    probabilities,
):
    """
    Calculate one-vs-rest ROC-AUC once for every class.

    Each column in binary_targets represents one class.
    Classes that contain only one target value are assigned NaN,
    because ROC-AUC is undefined for them.
    """

    binary_targets = _to_numpy(binary_targets)
    probabilities = _to_numpy(probabilities)

    if binary_targets.ndim != 2:
        raise ValueError(f"binary_targets must have shape [N,C]. Given: {binary_targets.shape}")

    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must have shape [N,C]. Given: {probabilities.shape}")

    if binary_targets.shape != probabilities.shape:
        raise ValueError(
            "binary_targets and probabilities must have the same shape. "
            f"Given: {binary_targets.shape} vs {probabilities.shape}"
        )

    roc_auc_values = []

    for class_index in range(binary_targets.shape[1]):
        class_targets = binary_targets[:, class_index]

        if len(np.unique(class_targets)) < 2:
            roc_auc_values.append(np.nan)
            continue

        roc_auc_values.append(
            float(
                roc_auc_score(
                    class_targets,
                    probabilities[:, class_index],
                )
            )
        )

    return roc_auc_values


def _safe_multilabel_roc_auc(
    targets,
    probabilities,
):
    """
    Calculate macro ROC-AUC for a multi-label task.

    Labels that contain only one target class are skipped,
    because ROC-AUC is undefined for them.
    """

    roc_auc_values = _safe_ovr_roc_auc_values(
        targets,
        probabilities,
    )

    valid_values = [value for value in roc_auc_values if not np.isnan(value)]

    if not valid_values:
        return np.nan

    return float(np.mean(valid_values))


# Genre - 19 Label Multi-Label Classification


def calculate_genre_probabilities(
    probabilities,
    targets,
    mask=None,
    threshold=0.5,
):
    """
    Calculate metrics for 19-label multi-label genre classification.

    Probabilities:
        [N,19] sigmoid probabilities for each independent label.

    Targets:
        [N,19] binary ground-truth labels.

    Mask:
        [N] validity mask.
        Only samples with mask value == 1 are evaluated.

    Threshold:
        Threshold used to convert probabilities into binary
        predictions.
    """

    probabilities = _to_numpy(probabilities)
    targets = _to_numpy(targets).astype(int)

    if probabilities.ndim != 2 or probabilities.shape[1] != 19:
        raise ValueError(
            f"Genre probabilities must have shape [N,19]. Given: {probabilities.shape}"
        )

    if targets.ndim != 2 or targets.shape[1] != 19:
        raise ValueError(f"Genre targets must have shape [N,19]. Given: {targets.shape}")

    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError(
            "Number of genre predictions and targets must match: "
            f"{probabilities.shape[0]} vs {targets.shape[0]}"
        )

    # Apply sample-level mask before evaluation.
    if mask is not None:
        probabilities = _apply_mask(probabilities, mask)
        targets = _apply_mask(targets, mask)

    # Convert probabilities into binary predictions.
    predictions = (probabilities >= threshold).astype(int)

    # Calculate one ROC-AUC value per genre exactly once.
    genre_roc_auc_values = _safe_ovr_roc_auc_values(
        targets,
        probabilities,
    )

    valid_genre_roc_auc_values = [value for value in genre_roc_auc_values if not np.isnan(value)]

    if valid_genre_roc_auc_values:
        macro_roc_auc = float(np.mean(valid_genre_roc_auc_values))
    else:
        macro_roc_auc = np.nan

    # Overall metrics.
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
        "macro_recall": float(
            recall_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_roc_auc": macro_roc_auc,
    }

    # Per-genre metrics.
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
            "roc_auc": genre_roc_auc_values[i],
        }

    return metrics


# Normal Multi-Class Classification


def calculate_classification_metrics(
    probabilities,
    targets,
    mask=None,
    class_names=None,
):
    """
    Calculate metrics for a multi-class classification task.

    Probabilities:
        [N,C] softmax probabilities.

    Targets:
        [N] integer class labels.

    Mask:
        [N] validity mask.
    """

    probabilities = _to_numpy(probabilities)
    targets = _to_numpy(targets).astype(int)

    # Check shapes.
    if probabilities.ndim != 2:
        raise ValueError(f"Probabilities must have shape [N,C]. Got {probabilities.shape}")

    if targets.ndim != 1:
        raise ValueError(f"Targets must have shape [N]. Got {targets.shape}")

    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Number of samples must match: {probabilities.shape[0]} vs {targets.shape[0]}"
        )

    # Apply mask.
    probabilities = _apply_mask(probabilities, mask)
    targets = _apply_mask(targets, mask)

    # Highest probability = predicted class.
    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    num_classes = probabilities.shape[1]

    # Default class names.
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    if len(class_names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(class_names)}")

    # Overall classification metrics.
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
        "macro_recall": float(
            recall_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }

    # Calculate one one-vs-rest ROC-AUC value per class exactly once.
    multiclass_binary_targets = (targets[:, None] == np.arange(num_classes)[None, :]).astype(int)

    class_roc_auc_values = _safe_ovr_roc_auc_values(
        multiclass_binary_targets,
        probabilities,
    )

    valid_class_roc_auc_values = [value for value in class_roc_auc_values if not np.isnan(value)]

    if valid_class_roc_auc_values:
        metrics["macro_roc_auc"] = float(np.mean(valid_class_roc_auc_values))
    else:
        metrics["macro_roc_auc"] = np.nan

    # Per-class metrics.
    report = classification_report(
        targets,
        predictions,
        labels=np.arange(num_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics["per_class"] = {}

    for class_index, class_name in enumerate(class_names):
        metrics["per_class"][f"class_{class_name}"] = {
            "f1": float(report[class_name]["f1-score"]),
            "precision": float(report[class_name]["precision"]),
            "recall": float(report[class_name]["recall"]),
            "support": int(report[class_name]["support"]),
            "roc_auc": class_roc_auc_values[class_index],
        }

    return metrics


# Box Office - 4-Class Classification


def calculate_box_office_metrics(
    probabilities,
    targets,
    mask,
):
    """
    Metrics for 4-class box-office classification.

    Class mapping:
        0 = Flop
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


# Content Rating - 4-Class Classification


def calculate_content_rating_metrics(
    probabilities,
    targets,
    mask,
):
    """
    Metrics for 4-class content-rating classification.

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


def calculate_rating_metrics(
    predictions,
    targets,
    mask,
):
    """
    Metrics for rating regression.

    Output:
        MAE
        RMSE
    """

    predictions = _to_numpy(predictions).reshape(-1)
    targets = _to_numpy(targets).reshape(-1)

    if predictions.shape[0] != targets.shape[0]:
        raise ValueError(
            "Number of predictions and targets must match: "
            f"{predictions.shape[0]} vs {targets.shape[0]}"
        )

    # Apply mask.
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
    mask=None,
    class_names=None,
):
    """
    Return predicted labels, confusion matrix,
    and classification report.
    """

    probabilities = _to_numpy(probabilities)
    targets = _to_numpy(targets).astype(int)

    if probabilities.ndim != 2:
        raise ValueError(f"Probabilities must have shape [N,C]. Got {probabilities.shape}")

    if targets.ndim != 1:
        raise ValueError(f"Targets must have shape [N]. Got {targets.shape}")

    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Number of samples must match: {probabilities.shape[0]} vs {targets.shape[0]}"
        )

    # Apply mask.
    probabilities = _apply_mask(
        probabilities,
        mask,
    )
    targets = _apply_mask(
        targets,
        mask,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    num_classes = probabilities.shape[1]

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    if len(class_names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(class_names)}")

    return {
        "predictions": predictions,
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=np.arange(num_classes),
        ),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=np.arange(num_classes),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


# Composite Score


def compute_composite_score(
    genre_macro_f1,
    rating_mae,
    box_office_macro_f1,
    content_rating_macro_f1,
    rating_max_error,
):
    """
    Calculate a single higher-is-better score
    for model selection.

    Classification metrics are already in [0,1].

    Rating MAE is converted into a score in [0,1]:

        rating_score =
            max(0, 1 - rating_mae / rating_max_error)

    All four tasks receive equal weight.

    ROC-AUC is not included in this composite score.
    """

    if rating_max_error <= 0:
        raise ValueError("rating_max_error must be greater than 0.")

    rating_score = max(
        0.0,
        1.0 - (rating_mae / rating_max_error),
    )

    return float(
        (genre_macro_f1 + rating_score + box_office_macro_f1 + content_rating_macro_f1) / 4.0
    )
