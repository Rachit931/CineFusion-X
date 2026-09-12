import mlflow
import torch

from src.evaluation.metrics import (
    calculate_box_office_metrics,
    calculate_content_rating_metrics,
    calculate_genre_probabilities,
    calculate_rating_metrics,
    compute_composite_score,
)
from src.losses.model_losses import MultiTaskLoss
from src.models.cinefusion_model import CineFusionModel

# DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PHASE 1 TRAINING


def train_phase_1(
    train_loader,
    val_loader,
    tabular_input_dim,
    learning_rate,
    epochs,
    tabular_hidden_dim,
    embedding_dim,
    rating_max_error,
    parameter_path=None,
    return_full_metrics=None,
):
    """
    Train one Phase-1 configuration on one CV fold.

    The train_loader and val_loader are created in
    cross_validation.py.

    Phase-1 objective:
        L_phase1 = L_task

    Returns: float or dict
        If return_full_metrics=None:
            best_composte_score

        If return_full_metrics not None:
            returnes best_metrics alongside best_composite_score
    """

    # Model
    model = CineFusionModel(
        tabular_input_dim=tabular_input_dim,
        tabular_hidden_dim=tabular_hidden_dim,
        embedding_dim=embedding_dim,
    )

    model = model.to(DEVICE)

    # LOSS
    criterion = MultiTaskLoss()

    # OPTIMIZER
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    # MLflow Parameters
    mlflow.log_params(
        {
            "tabular_input_dim": tabular_input_dim,
            "device": str(DEVICE),
            "rating_max_error": rating_max_error,
            "return_full_metrics": return_full_metrics,
        }
    )

    # BEST VALIDATION LOSS & BEST EPOCH TRACKING
    best_composite_score = float("-inf")
    best_epoch = 0

    # EPOCH LOOP
    for epoch in range(epochs):
        # Training
        model.train()

        running_train_loss = 0.0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            features = batch["features"].to(DEVICE)

            targets = {
                "genre": batch["genre_target"].to(DEVICE),
                "rating": batch["rating_target"].to(DEVICE),
                "box_office": batch["box_office_target"].to(DEVICE),
                "content_rating": batch["content_rating_target"].to(DEVICE),
            }

            masks = {
                "genre": batch["genre_mask"].to(DEVICE),
                "rating": batch["rating_mask"].to(DEVICE),
                "box_office": batch["box_office_mask"].to(DEVICE),
                "content_rating": batch["content_rating_mask"].to(DEVICE),
            }

            # Clear old gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                features=features,
            )

            # Calculate the multitasked masked loss
            losses = criterion(
                predictions=outputs["predictions"],
                targets=targets,
                masks=masks,
            )

            total_loss = losses["total_loss"]

            # Backpropogation
            total_loss.backward()

            # Update parameters
            optimizer.step()

            running_train_loss += total_loss.item()

        train_loss = running_train_loss / len(train_loader)

        # VALIDATION
        model.eval()

        running_val_loss = 0.0

        # Collecting predictions, targets and masks for all task heads
        # from the validation dataset for each fold.
        # In order to compute composite score.

        all_genre_probabilities = []
        all_genre_targets = []
        all_genre_masks = []

        all_rating_predictions = []
        all_rating_targets = []
        all_rating_masks = []

        all_box_office_probabilities = []
        all_box_office_targets = []
        all_box_office_masks = []

        all_content_rating_probabilities = []
        all_content_rating_targets = []
        all_content_rating_masks = []

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                features = batch["features"].to(DEVICE)

                targets = {
                    "genre": batch["genre_target"].to(DEVICE),
                    "rating": batch["rating_target"].to(DEVICE),
                    "box_office": batch["box_office_target"].to(DEVICE),
                    "content_rating": batch["content_rating_target"].to(DEVICE),
                }

                masks = {
                    "genre": batch["genre_mask"].to(DEVICE),
                    "rating": batch["rating_mask"].to(DEVICE),
                    "box_office": batch["box_office_mask"].to(DEVICE),
                    "content_rating": batch["content_rating_mask"].to(DEVICE),
                }

                # Forward pass
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    features=features,
                )

                # Validation loss
                losses = criterion(
                    predictions=outputs["predictions"],
                    targets=targets,
                    masks=masks,
                )

                running_val_loss += losses["total_loss"].item()

                # CONVERTING THE MODEL OUTPTUS FOR EACH TASK INTO
                # PROBABILITIES FROM PREDICTIONS(LOGITS).

                # Genre: logits into sigmoid probabilities.
                genre_probabilities = torch.sigmoid(outputs["predictions"]["genre"])

                # Rating: logits remains unchanged
                rating_predictions = outputs["predictions"]["rating"]

                # Box-Office: logits into softmax probabilities.
                box_office_probabilities = torch.softmax(
                    outputs["predictions"]["box_office"],
                    dim=1,
                )

                # Content-Rating: logits into softmax probabilities
                content_rating_probabilities = torch.softmax(
                    outputs["predictions"]["content_rating"], dim=1
                )

                # STORING THE VALIDATION OUTPUTS

                all_genre_probabilities.append(genre_probabilities.detach().cpu())

                all_genre_targets.append(targets["genre"].detach().cpu())

                all_genre_masks.append(masks["genre"].detach().cpu())

                all_rating_predictions.append(rating_predictions.detach().cpu())

                all_rating_targets.append(targets["rating"].detach().cpu())

                all_rating_masks.append(masks["rating"].detach().cpu())

                all_box_office_probabilities.append(box_office_probabilities.detach().cpu())

                all_box_office_targets.append(targets["box_office"].detach().cpu())

                all_box_office_masks.append(masks["box_office"].detach().cpu())

                all_content_rating_probabilities.append(content_rating_probabilities.detach().cpu())

                all_content_rating_targets.append(targets["content_rating"].detach().cpu())

                all_content_rating_masks.append(masks["content_rating"].detach().cpu())

        # Average validation loss for a particular fold
        val_loss = running_val_loss / len(val_loader)

        # Combine complete validation fold from list to
        # tensor to get correct probabilities or logits(ratingts)
        # of each task for all the validation points from the batch points.
        genre_probabilities = torch.cat(
            all_genre_probabilities,
            dim=0,
        )

        genre_targets = torch.cat(
            all_genre_targets,
            dim=0,
        )

        genre_masks = torch.cat(
            all_genre_masks,
            dim=0,
        )

        rating_predictions = torch.cat(
            all_rating_predictions,
            dim=0,
        )

        rating_targets = torch.cat(
            all_rating_targets,
            dim=0,
        )

        rating_masks = torch.cat(
            all_rating_masks,
            dim=0,
        )

        box_office_probabilities = torch.cat(
            all_box_office_probabilities,
            dim=0,
        )

        box_office_targets = torch.cat(
            all_box_office_targets,
            dim=0,
        )

        box_office_masks = torch.cat(
            all_box_office_masks,
            dim=0,
        )

        content_rating_probabilities = torch.cat(
            all_content_rating_probabilities,
            dim=0,
        )

        content_rating_targets = torch.cat(
            all_content_rating_targets,
            dim=0,
        )

        content_rating_masks = torch.cat(
            all_content_rating_masks,
            dim=0,
        )

        # CALCULATE VALIDATION METRICS NECCESSARY
        # FOR COMPUTING COMPOSITE SCORE FOR EACH FOLD

        # Genre
        genre_metrics = calculate_genre_probabilities(
            probabilities=genre_probabilities,
            targets=genre_targets,
            mask=genre_masks,
        )

        # Rating
        rating_metrics = calculate_rating_metrics(
            predictions=rating_predictions,
            targets=rating_targets,
            mask=rating_masks,
        )

        # Content-Rating
        content_rating_metrics = calculate_content_rating_metrics(
            probabilities=content_rating_probabilities,
            targets=content_rating_targets,
            mask=content_rating_masks,
        )

        # Box-Office
        box_office_metrics = calculate_box_office_metrics(
            probabilities=box_office_probabilities,
            targets=box_office_targets,
            mask=box_office_masks,
        )

        # COMPOSITE SCORE

        composite_score = compute_composite_score(
            genre_macro_f1=genre_metrics["macro_f1"],
            rating_mae=rating_metrics["mae"],
            box_office_macro_f1=box_office_metrics["macro_f1"],
            content_rating_macro_f1=content_rating_metrics["macro_f1"],
            rating_max_error=rating_max_error,
        )

        # SAVE THE BEST EPOCH FOR THIS FOLD.
        # BASED ON THE COMPOSITE SCORE(NOT VAL_LOSS)

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_val_loss = val_loss
            best_epoch = epoch + 1

            # Saving all of the metrics for the best epoch
            # in the memory for each fold

            # And returned ONLY when return_full_metrics=True.
            # Meaning for the best config giving best composite score.

            best_metrics = {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "composite_score": best_composite_score,
                "genre": genre_metrics,
                "rating": rating_metrics,
                "box_office": box_office_metrics,
                "content_rating": content_rating_metrics,
            }

            # Save checkpoint

            if parameter_path is not None:
                torch.save(
                    {
                        "epoch": best_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "tabular_input_dim": tabular_input_dim,
                        "tabular_hidden_dim": tabular_hidden_dim,
                        "embedding_dim": embedding_dim,
                        "learning_rate": learning_rate,
                    },
                    parameter_path,
                )

        # MLFLOW EPOCH LOGGING

        # Detailed metrics logged
        # ONLY for the best configuration achieved.

        if return_full_metrics:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "composite_score": composite_score,
                },
                step=epoch + 1,
            )

        # CONFIRMATION OUTPUT
        print(
            f"Epoch [{epoch + 1}/{epochs}]"
            f"Train Loss: {train_loss:.4f}"
            f"Val Loss: {val_loss:.4f}"
            f"Composite: {composite_score:.4f}"
        )

    # SAFETY CHECK
    if best_metrics is None:
        raise RuntimeError("No valid best epoch was found during training")

    # MLFLOW SUMMARY

    # Logging once (the best epoch)
    # for each fold of each config

    mlflow.log_metrics(
        {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_composite_score": best_composite_score,
        }
    )

    # DETAILED COMPLETE METRICS

    # ONLY logging them for the best selected configuration

    if return_full_metrics:
        # OVERALL METRICS

        mlflow.log_metrics(
            {
                # Genre
                "best_genre_macro_f1": best_metrics["genre"]["macro_f1"],
                "best_genre_micro_f1": best_metrics["genre"]["micro_f1"],
                "best_genre_macro_precision": best_metrics["genre"]["macro_precision"],
                # Rating
                "best_rating_mae": best_metrics["rating"]["mae"],
                "best_rating_rmse": best_metrics["rating"]["rmse"],
                # Box Office
                "best_box_office_macro_f1": best_metrics["box_office"]["macro_f1"],
                "best_box_office_weighted_f1": best_metrics["box_office"]["weighted_f1"],
                "best_box_office_accuracy": best_metrics["box_office"]["accuracy"],
                "best_box_office_balanced_accuracy": best_metrics["box_office"][
                    "balanced_accuracy"
                ],
                "best_box_office_macro_precision": best_metrics["box_office"]["macro_precision"],
                # Content Rating
                "best_content_rating_macro_f1": best_metrics["content_rating"]["macro_f1"],
                "best_content_rating_weighted_f1": best_metrics["content_rating"]["weighted_f1"],
                "best_content_rating_accuracy": best_metrics["content_rating"]["accuracy"],
                "best_content_rating_balanced_accuracy": best_metrics["content_rating"][
                    "balanced_accuracy"
                ],
                "best_content_rating_macro_precision": best_metrics["content_rating"][
                    "macro_precision"
                ],
            }
        )

        # PER-GENRE METRICS

        genre_metrics = {}

        for genre_name, genre_values in best_metrics["genre"]["per_genre"].items():
            genre_metrics[f"{genre_name}_f1"] = genre_values["f1"]
            genre_metrics[f"{genre_name}_precision"] = genre_values["precision"]
            genre_metrics[f"{genre_name}_recall"] = genre_values["recall"]

        mlflow.log_metrics(genre_metrics)

        # BOX OFFICE PER-CLASS METRICS

        box_office_metrics = {}

        for class_name, class_values in best_metrics["box_office"]["per_class"].items():
            box_office_metrics[f"{class_name}_f1"] = class_values["f1"]
            box_office_metrics[f"{class_name}_precision"] = class_values["precision"]
            box_office_metrics[f"{class_name}_recall"] = class_values["recall"]
            box_office_metrics[f"{class_name}_support"] = class_values["support"]

        mlflow.log_metrics(box_office_metrics)

        # CONTENT RATING PER-CLASS METRICS

        content_rating_metrics = {}

        for class_name, class_values in best_metrics["content_rating"]["per_class"].items():
            content_rating_metrics[f"{class_name}_f1"] = class_values["f1"]
            content_rating_metrics[f"{class_name}_precision"] = class_values["precision"]
            content_rating_metrics[f"{class_name}_recall"] = class_values["recall"]
            content_rating_metrics[f"{class_name}_support"] = class_values["support"]

    # CHECKPOINT ARTIFACT

    if parameter_path is not None:
        mlflow.log_artifact(str(parameter_path), artifact_path="parameters")

    # RETURNING the details metrics (NOT per-class)
    # for the best conifugration

    if return_full_metrics:
        return best_metrics

    return best_composite_score
