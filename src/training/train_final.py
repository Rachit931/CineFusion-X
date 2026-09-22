import json

import torch
from torch.utils.data import DataLoader

import src.utils as utils
from config.paths import (
    GENERAL_DIR,
    METRICS_DIR,
    MODEL_CONFIG_DIR,
    PARAMETERS_DIR,
    PHASE1_CACHE_DIR,
)
from src.dataset.c_model_data.custom_dataset import MovieDataset
from src.losses.model_losses import MultiTaskLoss
from src.models.cinefusion_model import CineFusionModel

# DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PATHS

MASTER_TRAIN = GENERAL_DIR / "master_training.csv"

BEST_CONFIG = MODEL_CONFIG_DIR / "phase1_best_config.json"
CV_RESULTS = METRICS_DIR / "phase1_cv_results.json"

FINAL_WEIGHTS = PARAMETERS_DIR / "phase1_final_weights.pt"

# TRAINING CONFIGURATION

BATCH_SIZE = 16
NUM_WORKERS = 4


def load_json(path):
    """
    Load a JSON file.
    """

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with open(path, encoding="utf-8") as file:
        return json.load(file)


def create_final_dataset():
    """
    Create the complete development dataset using
    the Phase-1 cached ViT and BERT representations
    """

    dataset = MovieDataset(
        master_path=MASTER_TRAIN,
        vit_image_transform=None,
        bert_tokenizer=None,
        poster_dir=None,
        max_text_length=256,
        cache_mode="phase1_cache",
        cache_dir=PHASE1_CACHE_DIR,
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


def train_final():
    """
    Train the final Phase-1 model on the complete
    development dataset.

    The hyperparameters are obtained through optuna
    by finding the best configuration

    The number of training epochs must comes from the
    mean best epoch observed during cross-validation.
    """

    # LOAD SELECTED CONFIGURATION

    best_config = load_json(BEST_CONFIG)
    cv_results = load_json(CV_RESULTS)

    required_config_keys = {
        "learning_rate",
        "weight_decay",
        "tabular_hidden_dim",
        "embedding_dim",
    }

    missing_keys = required_config_keys - best_config.keys()

    if missing_keys:
        raise KeyError(f"Missing keys in {BEST_CONFIG}: {sorted(missing_keys)}")

    if "mean_best_epoch" not in cv_results:
        raise KeyError(f"'mean_best_epoch' not found in {CV_RESULTS}")

    # FINAL EPOCH BUDGET

    mean_best_epoch = float(cv_results["mean_best_epoch"])

    final_epochs = max(1, int(round(mean_best_epoch)))

    # DATASET

    dataset = create_final_dataset()

    tabular_input_dim = dataset.features.shape[1]

    train_loader = create_final_loader(dataset)

    # MODEL

    model = CineFusionModel(
        tabular_input_dim=tabular_input_dim,
        tabular_hidden_dim=best_config["tabular_hidden_dim"],
        embedding_dim=best_config["embedding_dim"],
        trainable_vit_blocks=0,
        trainable_bert_layers=0,
        cache_mode="phase1_cache",
    ).to(DEVICE)

    # LOSS

    criterion = MultiTaskLoss().to(DEVICE)

    # OPTIMIZER

    trainable_parameters = []

    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=best_config["learning_rate"],
        weight_decay=best_config["weight_decay"],
    )

    # INFO
    utils.print_section("System Phase-1 Final Training")

    print(f"Device: {DEVICE}")
    print(f"Development samples: {len(dataset):,}")
    print(f"Tabular input dim: {tabular_input_dim}")
    print(f"Embedding dim: {best_config['embedding_dim']}")
    print(f"Tabular hidden dim: {best_config['tabular_hidden_dim']}")
    print(f"Learning rate: {best_config['learning_rate']:.3e}")
    print(f"Weight decay: {best_config['weight_decay']:.3e}")
    print(f"Mean best epoch: {mean_best_epoch:.2f}")
    print(f"Final training epochs: {final_epochs}")
    print()

    # FINAL TRAINING

    model.train()

    for epoch in range(final_epochs):
        running_loss = 0.0

        for batch_idx, batch in enumerate(
            train_loader,
            start=1,
        ):
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

            # Clearing the old gradients
            optimizer.zero_grad(set_to_none=True)

            # BF16 mixed-precision training
            with torch.autocast(
                device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"
            ):
                outputs = model(
                    features=features,
                    cached_visual_embedding=cached_visual_embedding,
                    cached_text_embedding=cached_text_embedding,
                )

                losses = criterion(predictions=outputs["predictions"], targets=targets, masks=masks)

                total_loss = losses["total_loss"]

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

        print(f"\rEpoch {epoch + 1}/{final_epochs} | Average Loss: {average_loss:.4f}")

    # SAVE FINAL CHECKPOINT

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": best_config,
        "tabular_input_dim": tabular_input_dim,
        "final_epochs": final_epochs,
        "mean_best_epoch": mean_best_epoch,
        "cache_mode": "phase1_cache",
    }

    torch.save(checkpoint, FINAL_WEIGHTS)

    print()
    print("Final Phase-1 training completed.")
    print("Checkpoint saved to:")
    print(FINAL_WEIGHTS)

    return FINAL_WEIGHTS


if __name__ == "__main__":
    train_final()
