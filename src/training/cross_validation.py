import random

import mlflow
import numpy as np
import torch
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Subset

import src.utils as utils
from src.training.training_config import train_phase_1

# CONFIGURATION

N_SPLITS = 3

BATCH_SIZE = 32
VAL_BATCH_SIZE = 128
NUM_WORKERS = 4

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# REPRODUCIBILITY


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# CROSS-VALIDATION


def cross_validate(
    config,
    rating_max_error,
    dataset,
    n_splits=N_SPLITS,
    seed=SEED,
):
    """
    Performing cross-validation for one configuration.
    The training data will be divided into time-ordered folds.

    For every fold:
    1. Create the train and validation subset.
    2. Create the Dataloaders of those subsets.
    3. Train on Phase-1 model.
    4. Record the validation loss for that fold.

    Test dataset will not be used.

    Ouptut :
        {
            "mean_composite_score": float,
            "std_composite_score": float,
            "mean_best_epoch": float,
            "fold_results": [
                {
                    "fold": int
                    "composite_score: float,
                    "best_epoch: float
                }
            ]
        }
    """

    # REPRODUCIBILITY
    set_seed(seed)

    # SAFETY CHECK
    if dataset.cache_mode != "phase1_cache":
        raise ValueError("Phase 1 cross-validation requires cache_mode='phase1_cache'.")

    if dataset.cache_split != "train":
        raise ValueError("Phase 1 cross-validation requires the train cache split.")

    # INPUT DIMENSION
    tabular_input_dim = dataset.features.shape[1]

    # CREATE THE ORDERED FOLDS

    splitter = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []

    # FOLD LOOP

    """
    1. splitter: Initializes the time based cross validation splitter.

    2. splitter(split(...)): For each fold, due to the time based
       splitting, there is no 80/20 splitting for each fold.
       It is assigning the indices of the dataset to training
       and validation data for each fold as per the time based splitting

       With time based: For the 1st fold splitting equal no. of indices will
       be divided to both based on time, and for each future fold,
       the number of indices keep moving towards training data based on the time.

    3. subset(...): assigns the points to the train dataset and validation dataset
       based on the indices assigned by the splitter(split(...)) for each fold.
    """

    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(range(len(dataset))),
        start=1,
    ):
        utils.print_section(f"PHASE-1 CROSS-VALIDATION - FOLD {fold}/{n_splits}")

        # CREATE FOLD DATASETS
        fold_train_dataset = Subset(
            dataset,
            train_indices,
        )

        fold_val_dataset = Subset(dataset, val_indices)

        # CREATE DATALOADERS FOR FOLDS
        fold_train_loader = DataLoader(
            fold_train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0),
        )

        fold_val_loader = DataLoader(
            fold_val_dataset,
            batch_size=VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0),
        )

        # MLflow FOLD RUN
        with mlflow.start_run(nested=True, run_name=f"fold_{fold}"):
            mlflow.log_params(
                {
                    "fold": fold,
                    "n_splits": n_splits,
                    "seed": seed,
                    "batch_size": BATCH_SIZE,
                    "device": str(DEVICE),
                    "cache_mode": dataset.cache_mode,
                }
            )

            # TRAIN ONE FOLD
            # For one fold: best composite score, best epoch, best val loss out of all the epoch

            fold_composite_score, best_epoch, val_loss_at_best_epoch = train_phase_1(
                train_loader=fold_train_loader,
                val_loader=fold_val_loader,
                tabular_input_dim=tabular_input_dim,
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                epochs=config["epochs"],
                tabular_hidden_dim=config["tabular_hidden_dim"],
                embedding_dim=config["embedding_dim"],
                rating_max_error=rating_max_error,
            )

            # SAFETY CHECK

            if not np.isfinite(fold_composite_score):
                raise RuntimeError(
                    f"Fold {fold} produced a non-finitecomposite_score: {fold_composite_score}"
                )

            if best_epoch <= 0:
                raise RuntimeError(f"Fold {fold} produced an invalid best epoch: {best_epoch}")

            # LOG FOLD RESULTS

            mlflow.log_metrics(
                {
                    "best_composite_score": float(fold_composite_score),
                    "best_epoch": int(best_epoch),
                    "best_val_loss": float(val_loss_at_best_epoch),
                }
            )

            fold_results.append(
                {
                    "fold": fold,
                    "composite_score": float(fold_composite_score),
                    "best_epoch": int(best_epoch),
                    "val_loss_at_best_epoch": float(val_loss_at_best_epoch),
                }
            )

        print(
            f"Fold {fold} complete | "
            f"Best Composite Score: {fold_composite_score:.4f} "
            f"Best Epoch: {best_epoch} "
            f"Best Val Loss At Best Epoch: {val_loss_at_best_epoch:.4f} "
        )

    # AGGREGATE CROSS-VALIDATION RESULTS

    composite_scores = []
    for result in fold_results:
        composite_scores.append(result["composite_score"])

    best_epochs = []
    for result in fold_results:
        best_epochs.append(result["best_epoch"])

    val_losses = []
    for result in fold_results:
        val_losses.append(result["val_loss_at_best_epoch"])

    mean_composite_score = float(np.mean(composite_scores))

    std_composite_score = float(np.std(composite_scores))

    mean_best_epoch = float(np.mean(best_epochs))

    mean_val_loss_at_best_epoch = float(np.mean(val_losses))

    # MLFLOW CONFIGURATION SUMMARY

    # At this point the nested fold run has ended,
    # so the active run is the parent Optuna trial run.

    if mlflow.active_run() is not None:
        mlflow.log_metrics(
            {
                "mean_cv_composite_score": mean_composite_score,
                "std_cv_composite_score": std_composite_score,
                "mean_best_epoch": mean_best_epoch,
                "mean_val_loss_at_best_epoch": mean_val_loss_at_best_epoch,
            }
        )

    # SUMMARY

    utils.print_section("CROSS VALIDATION SUMMARY")

    print(f"Mean CV Composite Score: {mean_composite_score:.4f}")

    print(f"Std CV Composite Score: {std_composite_score:.4f}")

    print(f"Mean Best Epoch: {mean_best_epoch:.2f}")

    print(f"Mean Val Loss: {mean_val_loss_at_best_epoch:.4f}")

    # RETURN

    return {
        "mean_composite_score": mean_composite_score,
        "std_composite_score": std_composite_score,
        "mean_best_epoch": mean_best_epoch,
        "mean_val_loss_at_best_epoch": mean_val_loss_at_best_epoch,
        "fold_results": fold_results,
    }
