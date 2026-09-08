import random

import mlflow
import numpy as np
import torch
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Subset

import src.utils as utils
from src.dataset.c_model_data.data_loader import train_dataset
from src.training.train_phase_1 import train_phase_1

# CONFIGURATION

N_SPLITS = 5

BATCH_SIZE = 20
NUM_WORKERS = 8

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
    dataset=train_dataset,
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
    """

    # REPRODUCIBILITY
    set_seed(seed)

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
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0),
        )

        # MLflow FOLD RUN
        with mlflow.start_run(nested=True, run_name=f"fold_{fold}"):
            mlflow.log_param("fold", fold)

            mlflow.log_param(
                "n_splits",
                n_splits,
            )

            mlflow.log_param(
                "seed",
                seed,
            )

            mlflow.log_param(
                "batch_size",
                BATCH_SIZE,
            )

            mlflow.log_param("device", str(DEVICE))

            # Logging the current configuration that is being evaluation for a particular fold
            mlflow.log_params(config)

            # TRAIN ONE FOLD

            result = train_phase_1(
                train_loader=fold_train_loader,
                val_loader=fold_val_loader,
                tabular_input_dim=tabular_input_dim,
                learning_rate=config["learning_rate"],
                epochs=config["epochs"],
                tabular_hidden_dim=config["tabular_hidden_dim"],
                embedding_dim=config["embedding_dim"],
                parameter_path=None,
            )

            # ACCUMULATING THE RESULTS

            fold_best_val_loss = result["best_val_loss"]
            fold_best_epoch = result["best_epoch"]

            mlflow.log_metric(
                "fold_best_val_loss",
                fold_best_val_loss,
            )

            mlflow.log_metric(
                "fold_best_epoch",
                fold_best_epoch,
            )

            fold_results.append(
                {
                    "fold": fold,
                    "best_val_loss": fold_best_val_loss,
                    "best_epoch": fold_best_epoch,
                }
            )

        print(f"Fold {fold} complete | Best Val Loss: {fold_best_val_loss: 4f}")

    # CONTAINGING THE BEST VALIDATION LOSS RESULTS
    # ACROSS EACH FOLD IN DIFFERENT DTYPES

    validation_losses = [result["best_val_loss"] for result in fold_results]

    mean_val_loss = float(np.mean(validation_losses))

    std_val_loss = float(np.std(validation_losses))

    best_epochs = [result["best_epoch"] for result in fold_results]

    mean_best_epoch = float(np.mean(best_epochs))

    # SUMMARY
    utils.print_section("CROSS VALIDATION SUMMARY")

    print(f"Mean Validation Loss: {mean_val_loss: 4f}")
    print(f"Std of Validation Loss: {std_val_loss: 4f}")
    print(f"Mean Best Epoch: {mean_best_epoch: 4f}")

    return {
        "mean_val_loss": mean_val_loss,
        "std_val_loss": std_val_loss,
        "mean_best_epoch": mean_best_epoch,
        "fold_results": fold_results,
    }
