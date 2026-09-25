import torch

from src.evaluation.metrics import (
    calculate_box_office_metrics,
    calculate_content_rating_metrics,
    calculate_genre_probabilities,
    calculate_rating_metrics,
    compute_composite_score,
)
from src.losses.model_losses import MultiTaskLoss
from src.models.movio_model import CineFusionModel

# DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINABLE_VIT_BLOCKS = 3
TRAINABLE_BERT_LAYERS = 3

# PHASE 1 TRAINING
# phase1_cache
# task losses only

# PHASE 2 TRAINING
# phase2_cache
# 3 trainable ViT blocks
# 3 trainable BERT layers
# task losses
# multimodal contrastive loss
# learnable contrastive_loss weighting

# Phase 2 can optionally intialize from a Phase-1 checkpoint.


def training(
    train_loader,
    val_loader,
    tabular_input_dim,
    learning_rate,
    weight_decay,
    epochs,
    tabular_hidden_dim,
    embedding_dim,
    rating_max_error,
    tabular_dropout,
    attention_dropout,
    contrastive_temp,
    phase="phase1",
):
    """
    Train one Phase-1 configuration on one CV fold.

    The train_loader and val_loader are created in
    cross_validation.py.

    Phase-1 objective:
        L_phase1 = L_task

    Phase-2 objective:
        L_phase2 = L_task + exp(-s) * L_infoNCE + s

    Returns: tuple
        best_composite_score, best_epoch, val_loss_at_best_epoch
    """
    # VALIDATE PHASE

    phase = phase.lower()

    if phase not in {"phase1", "phase2"}:
        raise ValueError("pahse must be either 'phase1' or 'phase2'.")

    # CACHE MODE

    if phase == "phase1":
        model = CineFusionModel(
            tabular_input_dim=tabular_input_dim,
            tabular_hidden_dim=tabular_hidden_dim,
            embedding_dim=embedding_dim,
            tabular_dropout=tabular_dropout,
            attention_dropout=attention_dropout,
            cache_mode="phase1_cache",
        )

    else:
        model = CineFusionModel(
            tabular_input_dim=tabular_input_dim,
            tabular_hidden_dim=tabular_hidden_dim,
            embedding_dim=embedding_dim,
            trainable_vit_blocks=TRAINABLE_VIT_BLOCKS,
            trainable_bert_layers=TRAINABLE_BERT_LAYERS,
            tabular_dropout=tabular_dropout,
            attention_dropout=attention_dropout,
            cache_mode="phase1_cache",
        )

    model = model.to(DEVICE)

    # LOSS

    if phase == "phase1":
        criterion = MultiTaskLoss().to(DEVICE)

    else:
        criterion = MultiTaskLoss(phase="phase2", contrastive_temperature=contrastive_temp).to(
            DEVICE
        )

    # OPTIMIZER
    trainable_parameters = []

    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)

    # In Phase 2, the loss contains the learnable
    # uncertainity parameter s.

    if phase == "phase2":
        for parameter in criterion.parameters():
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
    early_stopping_patience = 10
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
            # PHASE 1 INPUTS:

            if phase == "phase1":
                cached_visual_embedding = batch["cached_visual_embedding"].to(DEVICE)
                cached_text_embedding = batch["cached_text_embedding"].to(DEVICE)
                attention_mask = None

            # PHASE 2 INPUTS:

            else:
                cached_visual_embedding = None
                cached_text_embedding = None

                cached_visual_features = batch["cached_visual_features"].to(DEVICE)

                cached_text_features = batch["cached_text_features"].to(DEVICE)

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
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            # Using Mixed Precision training with FP16/BF16 instead of purely FP32
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.bfloat16,
                enabled=DEVICE.type == "cuda",
            ):
                # PHASE 1 FORWARD

                if phase == "phase1":
                    outputs = model(
                        features=features,
                        cached_visual_embedding=cached_visual_embedding,
                        cached_text_embedding=cached_text_embedding,
                    )

                # PHASE 2 FORWARD
                if phase == "phase2":
                    outputs = model(
                        features=features,
                        cached_visual_features=cached_visual_features,
                        cached_text_features=cached_text_features,
                        attention_mask=attention_mask,
                    )

                # Calculate the multitasked masked loss.

                # For Phase 1 this is only the four supervised
                # task losses.

                # For Phase 2 this also receives the three modality
                # embeddings required for the contrastive loss.
                if phase == "phase1":
                    losses = criterion(
                        predictions=outputs["predictions"],
                        targets=targets,
                        masks=masks,
                    )

                else:
                    losses = criterion(
                        predictions=outputs["predictions"],
                        targets=targets,
                        masks=masks,
                        visual_embedding=outputs["visual_embedding"],
                        text_embedding=outputs["text_embedding"],
                        tabular_embedding=outputs["tabular_embedding"],
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
                # PHASE 1 INPUTS

                if phase == "phase1":
                    cached_visual_embedding = batch["cached_visual_embedding"].to(DEVICE)
                    cached_text_embedding = batch["cached_text_embedding"].to(DEVICE)
                    attention_mask = None

                # PHASE 2 INPUTS
                else:
                    cached_visual_features = batch["cached_visual_features"].to(DEVICE)
                    cached_text_features = batch["cached_text_features"].to(DEVICE)
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
                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=torch.bfloat16,
                    enabled=DEVICE.type == "cuda",
                ):
                    # PHASE 1 FORWARD

                    if phase == "phase1":
                        outputs = model(
                            features=features,
                            cached_visual_embedding=cached_visual_embedding,
                            cached_text_embedding=cached_text_embedding,
                        )

                    # PHASE 2 FORWARD

                    else:
                        outputs = model(
                            features=features,
                            cached_visual_features=cached_visual_features,
                            cached_text_features=cached_text_features,
                            attention_mask=attention_mask,
                        )

                    # Validation loss

                    if phase == "phase1":
                        losses = criterion(
                            predictions=outputs["predictions"],
                            targets=targets,
                            masks=masks,
                        )

                    else:
                        losses = criterion(
                            predictions=outputs["predictions"],
                            targets=targets,
                            masks=masks,
                            visual_embedding=outputs["visual_embedding"],
                            text_embedding=outputs["text_embedding"],
                            tabular_embedding=outputs["tabular_embedding"],
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
            f"Epoch [{epoch + 1}/{epochs}] "
            f" Train Loss: {train_loss:.4f} "
            f" Val Loss: {val_loss:.4f} "
            f" Composite: {composite_score:.4f} "
            f" LR: {current_learning_rate:.6g} "
        )

        # EARLY STOPPING

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs. Best epoch: {best_epoch}")
            break

    # RETURN

    return best_composite_score, best_epoch, val_loss_at_best_epoch
