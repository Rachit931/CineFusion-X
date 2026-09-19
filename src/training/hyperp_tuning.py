import json

import mlflow
import optuna
import torch

import src.utils as utils
from config.paths import (
    METRICS_DIR,
    MODEL_CONFIG_DIR,
)
from src.dataset.c_model_data.data_loader import train_dataset
from src.training.cross_validation import (
    BATCH_SIZE,
    N_SPLITS,
    NUM_WORKERS,
    SEED,
    cross_validate,
)

# DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OUTPUT FILES

BEST_CONFIG = MODEL_CONFIG_DIR / "phase1_best_config.json"

CV_RESULTS = METRICS_DIR / "phase1_cv_results.json"

# OPTUNE COFIGURATION

N_TRIALS = 27

RATING_MAX_ERROR = 10.0

MIN_EPOCHS = 100
MAX_EPOCHS = 175
EPOCH_STEP = 25

EMBEDDING_DIM = 256

# JSON UTILITY


def save_json(data, path):
    """
    Save a Ptyhon object as a JSON file.
    """

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


# HYPERPARMETER SEARCH SPACE


def suggest_config(trial):
    """
    Suggesting one Phase - 1 hyperparameter
    configuration for an Optuna trial.
    """

    config = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-5,
            1e-4,
            log=True,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-6,
            1e-2,
            log=True,
        ),
        "epochs": trial.suggest_int(
            "epochs",
            MIN_EPOCHS,
            MAX_EPOCHS,
            step=EPOCH_STEP,
        ),
        "tabular_hidden_dim": trial.suggest_categorical(
            "tabular_hidden_dim",
            [256, 512],
        ),
        "embedding_dim": EMBEDDING_DIM,
    }

    return config


# OPTUNA OBJECTIVE


def objective(trial, dataset=train_dataset):
    """
    Run one Optuna trial.

    Flow:

        Optuna configuration
                ↓
        Time-series cross-validation
                ↓
        Mean CV composite score
                ↓
        Return score to Optuna

    The mean CV composite score is the quantity
    optimized by Optuna.
    """

    config = suggest_config(trial)

    print(
        f"\nTrial {trial.number + 1}/{N_TRIALS} | "
        f"LR: {config['learning_rate']:.3e} | "
        f"Weight Decay: {config['weight_decay']:.3e} | "
        f"Epochs: {config['epochs']} | "
        f"Tabular Hidden Dim: {config['tabular_hidden_dim']} | "
        f"Embedding Dim: {config['embedding_dim']}"
    )

    # TRIAL MLFLOW RUN

    with mlflow.start_run(
        nested=True,
        run_name=f"trial_{trial.number}",
    ):
        # Log the hyperparameter being evaluated
        mlflow.log_params(
            {
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "epochs": config["epochs"],
                "tabular_hidden_dim": config["tabular_hidden_dim"],
            }
        )

        mlflow.log_param("trial_number", trial.number)

        # CROSS-VALIDATION

        cv_results = cross_validate(
            config=config,
            rating_max_error=RATING_MAX_ERROR,
            dataset=dataset,
            n_splits=N_SPLITS,
            seed=SEED,
        )

        # EXTRACT CV RESULTS

        mean_composite_score = cv_results["mean_composite_score"]

        std_composite_score = cv_results["std_composite_score"]

        mean_best_epoch = cv_results["mean_best_epoch"]

        mean_val_loss_at_best_epoch = cv_results["mean_val_loss_at_best_epoch"]

        print(
            f"Trial {trial.number + 1}/{N_TRIALS} complete | "
            f"Mean CV Composite: {mean_composite_score:.4f} | "
            f"Std: {std_composite_score:.4f} | "
            f"Mean Best Epoch: {mean_best_epoch:.2f} | "
            f"Mean Val Loss at Best Epoch: "
            f"{mean_val_loss_at_best_epoch}"
        )

        # STORE ADDITIONAL TRIAL CONFIGURATION

        trial.set_user_attr(
            "std_cv_composite_score",
            std_composite_score,
        )

        trial.set_user_attr(
            "mean_best_epoch",
            mean_best_epoch,
        )

        trial.set_user_attr(
            "mean_val_loss_at_best_epoch",
            mean_val_loss_at_best_epoch,
        )

        # OBJECTIVE VALUE

        return mean_composite_score


# MAIN PHASE - 1 HYPERPARAMETER TUNING


def run_phase1_hyperparameter_tuning(
    dataset=train_dataset,
):
    """
    Run the complete Phase-1 hyperparameter search.
    Flow:
        Optuna
            ↓
        Trial
            ↓
        3-fold time-series CV
            ↓
        Mean CV composite score
            ↓
        Best configuration
            ↓
        Save phase1_best_config.json
            ↓
        Save phase1_cv_results.json

    Final Phase-1 training is NOT performed here.
    It is owned by train_final.py.
    """

    utils.print_section("PHASE-1 HYPERPARAMETER TUNING")

    # OPTUNA STUDY

    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="Phase_1_eval")

    # TOP LEVEL MLFLOW RUN

    with mlflow.start_run(run_name="phase1_hyperparameter_tuning"):
        # GLOBAL SEARCH SETTINGS

        mlflow.log_params(
            {
                "n_trials": N_TRIALS,
                "n_splits": N_SPLITS,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "seed": SEED,
                "device": str(DEVICE),
                "rating_max_error": RATING_MAX_ERROR,
                "embedding_dim": EMBEDDING_DIM,
                "min_epochs": MIN_EPOCHS,
                "max_epochs": MAX_EPOCHS,
                "epoch_step": EPOCH_STEP,
            }
        )

        # RUN OPTUNA

        study.optimize(
            lambda trial: objective(
                trial,
                dataset=dataset,
            ),
            n_trials=N_TRIALS,
            gc_after_trial=True,
        )

        # GET BEST TRIAL

        best_trial = study.best_trial

        if best_trial.value is None:
            raise RuntimeError("Best Optuna tiral has no objective value.")

        best_cv_composite_score = float(best_trial.value)

        best_config = dict(best_trial.params)

        # Embedding_dim is fixed and cannot be tuned
        best_config["embedding_dim"] = EMBEDDING_DIM

        # Mean best epoch is stored as trial metadata
        mean_best_epoch = float(best_trial.user_attrs["mean_best_epoch"])

        mean_val_loss_at_best_epoch = float(best_trial.user_attrs["mean_val_loss_at_best_epoch"])

        # SAVING THE BEST CONFIGURATION

        save_json(best_config, BEST_CONFIG)

        mlflow.log_artifact(str(BEST_CONFIG), artifact_path="configs")

        # BUILD OPTUNA / CV HISTORY

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
            "best_cv_composite_score": best_cv_composite_score,
            "mean_best_epoch": mean_best_epoch,
            "mean_val_loss_at_best_epoch": mean_val_loss_at_best_epoch,
            "best_config": best_config,
            "trials": trial_results,
        }

        # SAVE CV RESULTS

        save_json(cv_results, CV_RESULTS)

        mlflow.log_artifact(str(CV_RESULTS), artifact_path="metrics")

        # BEST-TRIAL SUMMARY ON TOP-LEVEL RUN

        mlflow.log_metrics(
            {
                "best_cv_composite_score": best_cv_composite_score,
                "best_mean_epoch": mean_best_epoch,
                "best_mean_val_loss_at_best_epoch": mean_val_loss_at_best_epoch,
            }
        )

    # SUMMARY

    utils.print_section("BEST PHASE-1 CONFIGURATION")

    for key, value in best_config.items():
        print(f"{key}: {value}")

    print(f"Best CV Composite Score: {best_cv_composite_score:.4f}")

    print(f"Mean Best Epoch: {mean_best_epoch:.2f}")

    print(f"Mean Val Loss at Best Epoch: {mean_val_loss_at_best_epoch:.4f}")

    print(f"Best Config: {BEST_CONFIG}")

    print(f"CV Results: {CV_RESULTS}")

    # RETURN

    return {
        "best_config": best_config,
        "best_cv_composite_score": best_cv_composite_score,
        "mean_best_epoch": mean_best_epoch,
        "mean_val_loss_at_best_epoch": mean_val_loss_at_best_epoch,
        "best_config_path": str(BEST_CONFIG),
        "cv_results_path": str(CV_RESULTS),
    }


if __name__ == "__main__":
    run_phase1_hyperparameter_tuning(
        dataset=train_dataset,
    )
