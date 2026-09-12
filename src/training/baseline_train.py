import json

import mlflow
import torch
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Subset

from config.paths import METRICS_DIR, PARAMETERS_DIR
from src.dataset.c_model_data.data_loader import train_dataset
from src.training.train_phase_1 import train_phase_1

# Configuration

BATCH_SIZE = 20
NUM_WORKERS = 8
N_SPLITS = 5

# Baseline hyperparameters

BASELINE_CONFIG = {
    "learning_rate": 5e-5,
    "epochs": 400,
    "tabular_hidden_dim": 512,
    "embedding_dim": 256,
}

RATING_MAX_ERROR = 10.0

BASELINE_WEIGHTS = PARAMETERS_DIR / "phase1_baseline.pt"

BASELINE_METRICS = METRICS_DIR / "phase_baseline_metrics.json"

# Device

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_baseline_loaders(dataset):
    """
    Create the train/validation loaders for the baseline run.

    Uses the final TimeSeriesSplit in the same way as the
    final detailed Phase-1 for the best config.
    """

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)

    splits = list(splitter.split(range(len(dataset))))

    train_indices, val_indices = splits[-1]

    train_dataset = Subset(dataset, train_indices)

    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    return train_loader, val_loader


# Baseline experiment=


def run_phase1_baseline(dataset=train_dataset):
    """
    Run the fixed-configuration Phase-1 baseline.

    The baseline:
        1. Uses a fixed configuration.
        2. Creates the train/validation loaders.
        3. Runs train_phase_1().
        4. Logs the experiment to MLflow.
        5. Saves the baseline metrics.

    No hyperparameter tuning or cross-validation is performed.
    """

    train_loader, val_loader = create_baseline_loaders(dataset)

    tabular_input_dim = dataset.features.shape[1]

    with mlflow.start_run(run_name="phase1_baseline"):
        # Log experiment metadata

        mlflow.log_params(
            {
                "experiment": "phase1_baseline",
                "tabular_input_dim": tabular_input_dim,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "n_splits": N_SPLITS,
                "device": str(DEVICE),
                "rating_max_error": RATING_MAX_ERROR,
            }
        )

        # Log the fixed baseline configuration.
        mlflow.log_params(BASELINE_CONFIG)

        # training the baseline model

        best_metrics = train_phase_1(
            train_loader=train_loader,
            val_loader=val_loader,
            tabular_input_dim=tabular_input_dim,
            learning_rate=BASELINE_CONFIG["learning_rate"],
            epochs=BASELINE_CONFIG["epochs"],
            tabular_hidden_dim=BASELINE_CONFIG["tabular_hidden_dim"],
            embedding_dim=BASELINE_CONFIG["embedding_dim"],
            rating_max_error=RATING_MAX_ERROR,
            parameter_path=BASELINE_WEIGHTS,
            return_full_metrics=True,
        )

        # Save baseline metrics
        with open(
            BASELINE_METRICS,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                best_metrics,
                f,
                indent=4,
            )

        mlflow.log_artifact(str(BASELINE_METRICS), artifact_path="metrics")

        # SUMMARY

        print("\n" + "=" * 60)
        print("PHASE 1 BASELINE COMPLETE")
        print("=" * 60)

        print(f"Best epoch: {best_metrics['best_epoch']}")

        print(f"Best composite score: {best_metrics['best_composite_score']:.6f}")

        print(f"Best validation loss: {best_metrics['best_val_loss']:.6f}")

        print(f"Checkpoint: {BASELINE_WEIGHTS}")

        print(f"Metrics: {BASELINE_METRICS}")

        print("=" * 60)

    return best_metrics


if __name__ == "__main__":
    run_phase1_baseline()
