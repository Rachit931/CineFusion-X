import mlflow
import torch

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
    parameter_path=None,
):
    """
    Train one Phase-1 configuration on one CV fold.

    The train_loader and val_loader are created in
    cross_validation.py.

    Phase-1 objective:
        L_phase1 = L_task

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
            "learning_rate": learning_rate,
            "epochs": epochs,
            "tabular_input_dim": tabular_input_dim,
            "tabular_hidden_dim": tabular_hidden_dim,
            "embedding_dim": embedding_dim,
            "device": str(DEVICE),
        }
    )

    # BEST VALIDATION LOSS
    best_val_loss = float("inf")
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

        val_loss = running_val_loss / len(val_loader)

        # MLflow METRICS

        mlflow.log_metric(
            "train_loss",
            train_loss,
            step=epoch + 1,
        )

        mlflow.log_metric(
            "val_loss",
            val_loss,
            step=epoch + 1,
        )

        # SAVE THE BEST CHECKPOINT FOR THIS RUN

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            # Saving the best model parameters into the path when not none
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

        # CONFIRMATION OUTPUT
        print(f"Epoch [{epoch + 1}/{epochs}]Train Loss: {train_loss:.4f}Val Loss: {val_loss:.4f}")

    # FINAL MLflow logging

    mlflow.log_metric(
        "best_val_loss",
        best_val_loss,
    )

    mlflow.log_metric(
        "best_epoch",
        best_epoch,
    )

    if parameter_path is not None:
        mlflow.log_artifact(
            str(parameter_path),
            artifact_path="parameters",
        )

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
