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
    weight_decay,
    epochs,
    tabular_hidden_dim,
    embedding_dim,
    rating_max_error,
):
    """
    Train one Phase-1 configuration on one CV fold.

    The train_loader and val_loader are created in
    cross_validation.py.

    Phase-1 objective:
        L_phase1 = L_task

    Returns: tuple
        best_composte_score, best_epoch, val_loss_at_best_epoch
    """

    # Model
    model = CineFusionModel(
        tabular_input_dim=tabular_input_dim,
        tabular_hidden_dim=tabular_hidden_dim,
        embedding_dim=embedding_dim,
        cache_mode="phase1_cache",
    )

    model = model.to(DEVICE)

    # LOSS
    criterion = MultiTaskLoss()

    # OPTIMIZER
    trainable_parameters = []

    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # LERNING-RATE SCHEDULAR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    # EARLY STOPPING
    early_stopping_patience = 15
    epochs_without_improvement = 0

    # BEST VALIDATION LOSS & BEST EPOCH TRACKING
    best_composite_score = float("-inf")
    val_loss_at_best_epoch = float("inf")
    best_epoch = 0

    # EPOCH LOOP
    for epoch in range(epochs):
        # Training
        model.train()

        running_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, start=1):
            cached_visual_embedding = batch["cached_visual_embedding"].to(DEVICE)
            cached_text_embedding = batch["cached_text_embedding"].to(DEVICE)
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
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            # Using Mixed Precision training with FP16/BF16 instead of purely FP32
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.bfloat16,
                enabled=DEVICE.type == "cuda",
            ):
                outputs = model(
                    features=features,
                    cached_visual_embedding=cached_visual_embedding,
                    cached_text_embedding=cached_text_embedding,
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

            print(
                f"\rEpoch {epoch + 1}/{epochs} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {total_loss.item():.4f}",
                end="",
                flush=True,
            )

        print()

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

        with torch.inference_mode():
            for batch in val_loader:
                cached_visual_embedding = batch["cached_visual_embedding"].to(DEVICE)
                cached_text_embedding = batch["cached_text_embedding"].to(DEVICE)
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
                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=torch.bfloat16,
                    enabled=DEVICE.type == "cuda",
                ):
                    outputs = model(
                        features=features,
                        cached_visual_embedding=cached_visual_embedding,
                        cached_text_embedding=cached_text_embedding,
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

        # LEARNING-RATE SCHEDULER
        # Internally checking the composite score if have increased or not.
        scheduler.step(composite_score)

        # stores the current learning rate
        current_learning_rate = optimizer.param_groups[0]["lr"]

        # SAVE THE BEST EPOCH FOR THIS FOLD.
        # BASED ON THE COMPOSITE SCORE(NOT VAL_LOSS)

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            val_loss_at_best_epoch = val_loss
            best_epoch = epoch + 1

            # Reset early-stopping counter because the model improved.
            epochs_without_improvement = 0

        else:
            # No improvement in validation composite score.
            epochs_without_improvement += 1

        # CONFIRMATION OUTPUT
        print(
            f"Epoch [{epoch + 1}/{epochs}]"
            f"Train Loss: {train_loss:.4f}"
            f"Val Loss: {val_loss:.4f}"
            f"Composite: {composite_score:.4f}"
            f"LR: {current_learning_rate:.6g}"
        )

        # EARLY STOPPING

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs. Best epoch: {best_epoch}")
            break

    # RETURN

    return best_composite_score, best_epoch, val_loss_at_best_epoch
