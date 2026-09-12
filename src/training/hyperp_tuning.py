import json

import mlflow
import optuna
import torch
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Subset

import src.utils as utils
from config.paths import (
    METRICS_DIR,
    MODEL_CONFIG_DIR,
    PARAMETERS_DIR,
)
from src.dataset.c_model_data.data_loader import train_dataset
from src.training.cross_validation import (
    BATCH_SIZE,
    N_SPLITS,
    NUM_WORKERS,
    SEED,
    cross_validate,
)
from src.training.train_phase_1 import train_phase_1

# DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OUTPUT FILES

BEST_CONFIG = MODEL_CONFIG_DIR / "phase1_best_config.json"

BEST_METRICS = METRICS_DIR / "phase1_best_metrics.json"

BEST_WEIGHTS = PARAMETERS_DIR / "phase1_best.pt"

CV_RESULTS = METRICS_DIR / "phase1_cv_results.json"

# OPTUNE COFIGURATION

N_TRIALS = 37

RATING_MAX_ERROR = 10.0

# SAVING JSONs


def save_json(data, path):
    """
    Save a Python dictionary as a JSON file.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def suggest_config(trial):
    """
    Suggest one Phase-1 hyperparameter configuration
    for Optuna trial.

    Outputs:
        dict:
            Hyperparameter configuration.
    """

    config = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-5,
            1e-4,
            log=True,
        ),
        "epochs": trial.suggest_int(
            "epochs",
            300,
            400,
            step=50,
        ),
        "tabular_hidden_dim": trial.suggest_categorical("tabular_hidden_dim", [256, 512]),
        "embedding_dim": trial.suggest_categorical(
            "embedding_dim",
            [256],
        ),
    }

    return config


# OBJECTIVE FUNCTION


def objective(
    trial,
    dataset=train_dataset,
):
    """
    Optuna objective.

    For each trial:
        * Suggest a configuration.
        * Run 5-fold time-series CV..
        * Obtain mean CV composite score.
        * Return that score to Optuna.

    Only the mean CV composite score is optimized.

    Detailed metrics are NOT generated for every trial.
    """

    config = suggest_config(trial)

    # Parent trial run

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        # Log trial parameters

        mlflow.log_params(config)

        mlflow.log_param(
            "trial_number",
            trial.number,
        )

        # Run cross-validation

        cv_results = cross_validate(
            config=config,
            rating_max_error=RATING_MAX_ERROR,
            dataset=dataset,
            n_splits=N_SPLITS,
            seed=SEED,
        )

        # Extract optimization objective

        mean_composite_score = cv_results["mean_composite_score"]

        std_composite_score = cv_results["std_composite_score"]

        # Report objective to Optune

        trial.set_user_attr("std_cv_composite_score", std_composite_score)

        return mean_composite_score


# FINAL BEST-CONFIG LOADERS
def create_final_best_config_loader(
    dataset,
):
    """
    Create train and validation loaders for the final
    complete detailed Phase-1 run.

    Uses the final TimeSeries SPlit fold.

    Creating the train and validation loader and taking
    the final time series split through CV to run on the
    best selected config to get the detailed metrics
    """

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)

    splits = list(splitter.split(range(len(dataset))))

    train_indices, val_indices = splits[-1]

    # Assinging the training and validation datasets
    # according to the final split

    fold_train_dataset = Subset(
        dataset,
        train_indices,
    )

    fold_val_dataset = Subset(
        dataset,
        val_indices,
    )

    train_loader = DataLoader(
        fold_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        fold_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    return train_loader, val_loader


# MAIN & FINAL PHASE - 1 PIPELINE


def run_phase1_hyperparameter_tuning(
    dataset=train_dataset,
):
    """
    Complete Phase-1 Pipeline.

    Flow:

        Optuna
            ↓
        Trial
            ↓
        5-fold CV
            ↓
        mean composite score
            ↓
        Best configuration
            ↓
        Save best config
            ↓
        Detailed best-config training
            ↓
        Save best metrics
            ↓
        Save best checkpoint
            ↓
        Save CV/Optuna results
    """

    utils.print_section("PHASE-1 HYPERPARAMETER TUNING")

    # OPTUNA STUDY

    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="cinefusion_phase1",
    )

    # MAIN MLFLOW RUN

    with mlflow.start_run(run_name="phase1_hyperparameter_tuning"):
        # Global experiment settings

        mlflow.log_params(
            {
                "n_trials": N_TRIALS,
                "n_splits": N_SPLITS,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "seed": SEED,
                "device": str(DEVICE),
                "rating_max_error": RATING_MAX_ERROR,
            }
        )

        # RUN OPTUNA

        study.optimize(
            lambda trial: objective(trial, dataset=dataset),
            n_trials=N_TRIALS,
        )

        # GET BEST TRIAL

        best_trial = study.best_trial

        best_config = dict(best_trial.params)

        best_cv_composite_score = float(best_trial.value)

        # SAVE BEST CONFIG

        save_json(best_config, BEST_CONFIG)

        mlflow.log_artifact(str(BEST_CONFIG), artifact_path="configs")

        # BEST CONFIG SUMMARY

        utils.print_section("BEST PHASE-1 CONFIGURATION")

        for key, value in best_config.items():
            print(f"{key}: {value}")

        print(f"Best CV Composite Score: {best_cv_composite_score:.4f}")

        # FINAL DETAILED BEST-CONFIG RUN

        utils.print_section("FINAL PHASE-1 TRAINING")

        final_train_loader, final_val_loader = create_final_best_config_loader(dataset=dataset)

        with mlflow.start_run(nested=True, run_name="phase1_best_configuration"):
            # Best configuration is logged here once
            # alongside with the detailed metrics of it.
            # And CV_results(all runs) will also be saved.

            mlflow.log_params(best_config)

            mlflow.log_params(
                {
                    "best_trial_number": best_trial.number,
                    "best_cv_composite_score": best_cv_composite_score,
                }
            )

            # Train selected configuration in detailed

            best_metrics = train_phase_1(
                train_loader=final_train_loader,
                val_loader=final_train_loader,
                tabular_input_dim=dataset.features.shape[1],
                learning_rate=best_config["learning_rate"],
                epochs=best_config["epochs"],
                tabular_hidden_dim=best_config["tabular_hidden_dim"],
                embedding_dim=best_config["embedding_dim"],
                rating_max_error=RATING_MAX_ERROR,
                parameter_path=BEST_WEIGHTS,
                return_full_metrics=True,
            )

            # Save complete best-config metrics

            save_json(best_metrics, BEST_METRICS)

            mlflow.log_artifact(
                str(BEST_METRICS),
                artifact_path="metrics",
            )

        # SAVE ALL OPTUNA RESULTS

        trial_results = []

        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            trial_results.append(
                {
                    "trial_number": trial.number,
                    "value": (float(trial.value) if trial.value is not None else None),
                    "params": dict(trial.params),
                    "user_attrs": dict(trial.user_attrs),
                }
            )
        cv_results = {
            "best_trial_number": best_trial.number,
            "best_cv_composite_score": (best_cv_composite_score),
            "best_config": best_config,
            "trials": trial_results,
        }

        save_json(
            cv_results,
            CV_RESULTS,
        )

        mlflow.log_artifact(
            str(CV_RESULTS),
            artifact_path="metrics",
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    utils.print_section("PHASE-1 COMPLETE")

    print(f"Best CV Composite Score: {best_cv_composite_score:.4f}")

    print(f"Best Config: {BEST_CONFIG}")

    print(f"Best Metrics: {BEST_METRICS}")

    print(f"Best Checkpoint: {BEST_WEIGHTS}")

    print(f"CV Results: {CV_RESULTS}")

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "best_config": best_config,
        "best_cv_composite_score": (best_cv_composite_score),
        "best_metrics": best_metrics,
        "best_config_path": str(BEST_CONFIG),
        "best_metrics_path": str(BEST_METRICS),
        "best_checkpoint_path": str(BEST_WEIGHTS),
        "cv_results_path": str(CV_RESULTS),
    }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_phase1_hyperparameter_tuning(
        dataset=train_dataset,
    )
