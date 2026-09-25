import json

import mlflow
import torch
from torch.utils.data import DataLoader

import src.utils as utils
from config.paths import (
    GENERAL_DIR,
    METRICS_DIR,
    MODEL_CONFIG_DIR,
    PARAMETERS_DIR,
    PHASE1_CACHE_DIR,
    PHASE2_CACHE_DIR,
)
from src.dataset.c_model_data.custom_dataset import MovieDataset
from src.losses.model_losses import MultiTaskLoss
from src.models.movio_model import MovioModel
from src.training.training_config import (
    TRAINABLE_BERT_LAYERS,
    TRAINABLE_VIT_BLOCKS,
)

# DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PATHS

MASTER_TRAIN = GENERAL_DIR / "master_training.csv"

BEST_CONFIG = {
    "phase1": MODEL_CONFIG_DIR / "phase1_best_config.json",
    "phase2": MODEL_CONFIG_DIR / "phase2_best_config.json",
}

CV_RESULTS = {
    "phase1": METRICS_DIR / "phase1_cv_results.json",
    "phase2": METRICS_DIR / "phase2_cv_results.json",
}

FINAL_WEIGHTS = {
    "phase1": PARAMETERS_DIR / "phase1_final_weights.pt",
    "phase2": PARAMETERS_DIR / "phase2_final_weights.pt",
}

CACHE_DIRS = {
    "phase1": PHASE1_CACHE_DIR,
    "phase2": PHASE2_CACHE_DIR,
}


# TRAINING CONFIGURATION

BATCH_SIZE = 16
NUM_WORKERS = 4


# JSON UTILITY


def load_json(path):
    """
    Load a JSON file.
    """

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with open(
        path,
        encoding="utf-8",
    ) as file:
        return json.load(file)


# DATASET


def create_final_dataset(phase):
    """
    Create the complete development dataset using
    the cache corresponding to the selected phase.
    """

    cache_mode = f"{phase}_cache"

    dataset = MovieDataset(
        master_path=MASTER_TRAIN,
        vit_image_transform=None,
        bert_tokenizer=None,
        poster_dir=None,
        max_text_length=256,
        cache_mode=cache_mode,
        cache_dir=CACHE_DIRS[phase],
        cache_split="train",
    )

    return dataset


def create_final_loader(dataset):
    """
    Create the DataLoader for final development training.
    """

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )


# FINAL TRAINING


def train_final(phase):
    """
    Train the selected Phase-1 or Phase-2 configuration
    on the complete development dataset.

    The selected hyperparameters come from the best Optuna
    configuration.

    The final epoch budget comes from the mean best epoch
    observed during cross-validation.
    """

    # PHASE VALIDATION

    phase = phase.lower()

    if phase not in {"phase1", "phase2"}:
        raise ValueError("phase must be either 'phase1' or 'phase2'.")

    cache_mode = f"{phase}_cache"

    best_config_path = BEST_CONFIG[phase]
    cv_results_path = CV_RESULTS[phase]
    final_weights_path = FINAL_WEIGHTS[phase]

    # LOAD SELECTED CONFIGURATION

    best_config = load_json(best_config_path)

    cv_results = load_json(cv_results_path)

    required_config_keys = {
        "learning_rate",
        "weight_decay",
        "tabular_hidden_dim",
        "embedding_dim",
        "tabular_dropout",
        "attention_dropout",
    }

    if phase == "phase2":
        required_config_keys.add("contrastive_temperature")

    missing_keys = required_config_keys - best_config.keys()

    if missing_keys:
        raise KeyError(f"Missing keys in {best_config_path}: {sorted(missing_keys)}")

    if "mean_best_epoch" not in cv_results:
        raise KeyError(f"'mean_best_epoch' not found in {cv_results_path}")

    # FINAL EPOCH BUDGET

    mean_best_epoch = float(cv_results["mean_best_epoch"])

    final_epochs = max(
        1,
        int(round(mean_best_epoch)),
    )

    # DATASET

    dataset = create_final_dataset(phase=phase)

    tabular_input_dim = dataset.features.shape[1]

    train_loader = create_final_loader(dataset)

    # MODEL

    if phase == "phase1":
        trainable_vit_blocks = 0
        trainable_bert_layers = 0

    else:
        trainable_vit_blocks = TRAINABLE_VIT_BLOCKS
        trainable_bert_layers = TRAINABLE_BERT_LAYERS

    model = MovioModel(
        tabular_input_dim=tabular_input_dim,
        tabular_hidden_dim=best_config["tabular_hidden_dim"],
        embedding_dim=best_config["embedding_dim"],
        trainable_vit_blocks=trainable_vit_blocks,
        trainable_bert_layers=trainable_bert_layers,
        tabular_dropout=best_config["tabular_dropout"],
        attention_dropout=best_config["attention_dropout"],
        cache_mode=cache_mode,
    ).to(DEVICE)

    # LOSS

    if phase == "phase1":
        criterion = MultiTaskLoss(phase="phase1").to(DEVICE)

    else:
        criterion = MultiTaskLoss(
            phase="phase2",
            contrastive_temperature=best_config["contrastive_temperature"],
        ).to(DEVICE)

    # OPTIMIZER

    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    if not model_parameters:
        raise RuntimeError("No trainable model parameters found.")

    optimizer_parameter_groups = [
        {
            "params": model_parameters,
            "weight_decay": best_config["weight_decay"],
        }
    ]

    # Phase 2 learnable log-variance parameter.
    #
    # Do not apply AdamW weight decay to this parameter.
    if phase == "phase2":
        contrastive_parameters = [
            parameter for parameter in criterion.parameters() if parameter.requires_grad
        ]

        optimizer_parameter_groups.append(
            {
                "params": contrastive_parameters,
                "weight_decay": 0.0,
            }
        )

    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups,
        lr=best_config["learning_rate"],
    )

    # INFO

    utils.print_section(f"System {phase.upper()} Final Training")

    print(f"Device: {DEVICE}")
    print(f"Development samples: {len(dataset):,}")
    print(f"Tabular input dim: {tabular_input_dim}")
    print(f"Embedding dim: {best_config['embedding_dim']}")
    print(f"Tabular hidden dim: {best_config['tabular_hidden_dim']}")
    print(f"Tabular dropout: {best_config['tabular_dropout']:.2f}")
    print(f"Attention dropout: {best_config['attention_dropout']:.2f}")
    print(f"Learning rate: {best_config['learning_rate']:.3e}")
    print(f"Weight decay: {best_config['weight_decay']:.3e}")
    print(f"Mean best epoch: {mean_best_epoch:.2f}")
    print(f"Final training epochs: {final_epochs}")

    if phase == "phase2":
        print(f"Trainable ViT blocks: {trainable_vit_blocks}")
        print(f"Trainable BERT layers: {trainable_bert_layers}")
        print(f"Contrastive temperature: {best_config['contrastive_temperature']:.4f}")

    print()

    # MLFLOW FINAL TRAINING RUN

    with mlflow.start_run(run_name=f"{phase}_final_training"):
        # FINAL-RUN METADATA
        #
        # Do not log the complete tuning/search history again.
        # This run is only for the final selected model.

        mlflow.log_params(
            {
                "phase": phase,
                "cache_mode": cache_mode,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "device": str(DEVICE),
                "development_samples": len(dataset),
                "tabular_input_dim": tabular_input_dim,
                "final_epochs": final_epochs,
                "mean_best_epoch": mean_best_epoch,
                "trainable_vit_blocks": trainable_vit_blocks,
                "trainable_bert_layers": trainable_bert_layers,
            }
        )

        mlflow.set_tag(
            "best_config_path",
            str(best_config_path),
        )

        mlflow.set_tag(
            "cv_results_path",
            str(cv_results_path),
        )

        # FINAL TRAINING

        model.train()

        for epoch in range(final_epochs):
            running_loss = 0.0

            for batch_idx, batch in enumerate(
                train_loader,
                start=1,
            ):
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

                # PHASE-SPECIFIC INPUTS

                if phase == "phase1":
                    cached_visual_embedding = batch["cached_visual_embedding"].to(DEVICE)

                    cached_text_embedding = batch["cached_text_embedding"].to(DEVICE)

                else:
                    cached_visual_features = batch["cached_visual_features"].to(DEVICE)

                    cached_text_features = batch["cached_text_features"].to(DEVICE)

                    attention_mask = batch["attention_mask"].to(DEVICE)

                # CLEAR OLD GRADIENTS

                optimizer.zero_grad(set_to_none=True)

                # FORWARD + LOSS

                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=torch.bfloat16,
                    enabled=DEVICE.type == "cuda",
                ):
                    if phase == "phase1":
                        outputs = model(
                            features=features,
                            cached_visual_embedding=(cached_visual_embedding),
                            cached_text_embedding=(cached_text_embedding),
                        )

                        losses = criterion(
                            predictions=outputs["predictions"],
                            targets=targets,
                            masks=masks,
                        )

                    else:
                        outputs = model(
                            features=features,
                            cached_visual_features=(cached_visual_features),
                            cached_text_features=(cached_text_features),
                            attention_mask=attention_mask,
                        )

                        losses = criterion(
                            predictions=outputs["predictions"],
                            targets=targets,
                            masks=masks,
                            visual_embedding=outputs["visual_embedding"],
                            text_embedding=outputs["text_embedding"],
                            tabular_embedding=outputs["tabular_embedding"],
                        )

                    total_loss = losses["total_loss"]

                # BACKPROPAGATION

                total_loss.backward()

                optimizer.step()

                running_loss += total_loss.item()

                print(
                    f"\rEpoch {epoch + 1}/{final_epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {total_loss.item():.4f}",
                    end="",
                    flush=True,
                )

            average_loss = running_loss / len(train_loader)

            # MLFLOW EPOCH METRICS

            mlflow.log_metric(
                "train_loss",
                float(average_loss),
                step=epoch + 1,
            )

            if phase == "phase2":
                log_variance = float(criterion.log_variance_contrastive.detach().item())

                lambda_contrastive = float(
                    torch.exp(-criterion.log_variance_contrastive.detach()).item()
                )

                mlflow.log_metrics(
                    {
                        "log_variance_contrastive": (log_variance),
                        "lambda_contrastive": (lambda_contrastive),
                    },
                    step=epoch + 1,
                )

            print(f"\rEpoch {epoch + 1}/{final_epochs} | Average Loss: {average_loss:.4f}")

        # SAVE FINAL CHECKPOINT

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "config": best_config,
            "phase": phase,
            "cache_mode": cache_mode,
            "tabular_input_dim": tabular_input_dim,
            "final_epochs": final_epochs,
            "mean_best_epoch": mean_best_epoch,
            "trainable_vit_blocks": trainable_vit_blocks,
            "trainable_bert_layers": trainable_bert_layers,
        }

        torch.save(
            checkpoint,
            final_weights_path,
        )

        # LOG FINAL CHECKPOINT

        mlflow.log_artifact(
            str(final_weights_path),
            artifact_path="checkpoints",
        )

    print()
    print(f"Final {phase.upper()} training completed.")
    print("Checkpoint saved to:")
    print(final_weights_path)

    return final_weights_path


if __name__ == "__main__":
    train_final(phase="phase2")
