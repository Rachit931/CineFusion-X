import json

import mlflow
import optuna
import torch

import src.utils as utils
from config.paths import (
    METRICS_DIR,
    MODEL_CONFIG_DIR,
)
from src.dataset.c_model_data.data_loader import create_dataset
from src.training.cross_validation import (
    BATCH_SIZE,
    N_SPLITS,
    NUM_WORKERS,
    SEED,
    VAL_BATCH_SIZE,
    cross_validate,
)
from src.training.training_config import (
    TRAINABLE_BERT_LAYERS,
    TRAINABLE_VIT_BLOCKS,
)

# DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OUTPUT FILES

BEST_CONFIG = {
    "phase1": MODEL_CONFIG_DIR / "phase1_best_config.json",
    "phase2": MODEL_CONFIG_DIR / "phase2_best_config.json",
}

CV_RESULTS = {
    "phase1": METRICS_DIR / "phase1_cv_results.json",
    "phase2": METRICS_DIR / "phase2_cv_results.json",
}

# OPTUNE COFIGURATION

N_TRIALS = 27

RATING_MAX_ERROR = 10.0

MIN_EPOCHS = 100
MAX_EPOCHS = 200
EPOCH_STEP = 25

EMBEDDING_DIM = 256

MIN_DROPOUT = 0.0
MAX_DROPOUT = 0.5
DROPOUT_STEP = 0.05

MIN_CONTRASTIVE_TEMPERATURE = 0.03
MAX_CONTRASTIVE_TEMPERATURE = 0.20

# JSON UTILITY


def save_json(data, path):
    """
    Save a Ptyhon object as a JSON file.
    """

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


# HYPERPARMETER SEARCH SPACE


def suggest_config(trial, phase):
    """
    Suggesting hyperparameter
    configuration for an Optuna trial.

    Common hyperparameters:
        - learning-rate
        - weight_decay
        - epochs
        - tabular_hidden_dim
        - tabular_dropout
        - attention_dropout

    Phase 2 additionally tunes:
        -contrastive_tempearature
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
        "tabular_dropout": trial.suggest_float(
            "tabular_dropout",
            MIN_DROPOUT,
            MAX_DROPOUT,
            step=DROPOUT_STEP,
        ),
        "attention_dropout": trial.suggest_float(
            "attention_dropout",
            MIN_DROPOUT,
            MAX_DROPOUT,
            step=DROPOUT_STEP,
        ),
        "embedding_dim": EMBEDDING_DIM,
    }

    if phase == "phase2":
        config["contrastive_temperature"] = trial.suggest_float(
            "contrastive_temperature",
            MIN_CONTRASTIVE_TEMPERATURE,
            MAX_CONTRASTIVE_TEMPERATURE,
            log=True,
        )

    return config


# OPTUNA OBJECTIVE


def objective(trial, dataset, phase):
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

    config = suggest_config(
        trial=trial,
        phase=phase,
    )

    print(
        f"\nTrial {trial.number + 1}/{N_TRIALS} | "
        f"LR: {config['learning_rate']:.3e} | "
        f"Weight Decay: {config['weight_decay']:.3e} | "
        f"Epochs: {config['epochs']} | "
        f"Tabular Hidden Dim: {config['tabular_hidden_dim']} | "
        f"Embedding Dim: {config['embedding_dim']} | "
        f"Tabular Dropout: {config['tabular_dropout']:.2f} | "
        f"Attention Dropout: {config['attention_dropout']} | "
    )

    if phase == "phase2":
        print(f"Contrastive Temperature: {config['contrastive_temperature']:.4f}")

    # TRIAL MLFLOW RUN

    with mlflow.start_run(
        nested=True,
        run_name=f"trial_{trial.number}",
    ):
        # Log the hyperparameter being evaluated
        trial_params = {
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "epochs": config["epochs"],
            "tabular_hidden_dim": config["tabular_hidden_dim"],
            "tabular_dropout": config["tabular_dropout"],
            "attention_dropout": config["attention_dropout"],
        }

        if phase == "phase2":
            trial_params["contrastive_temperature"] = config["contrastive_temperature"]

        mlflow.log_params(trial_params)

        mlflow.log_param("trial_number", trial.number)

        # CROSS-VALIDATION

        cv_results = cross_validate(
            config=config,
            rating_max_error=RATING_MAX_ERROR,
            dataset=dataset,
            n_splits=N_SPLITS,
            seed=SEED,
            phase=phase,
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


def run_hyperparameter_tuning(phase):
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

    Phase 1:
        phase1_cache
        supervised multitask objective

    Phase 2:
        phase2_cache
        supervised multitask objective
        + multimodal contrastive objective
    """
    phase = phase.lower()

    if phase not in {"phase1", "phase2"}:
        raise ValueError("phase must be either 'phase1' or 'phase2'.")

    # PHASE-SPECIFIC PATHS

    best_config_path = BEST_CONFIG[phase]
    cv_results_path = CV_RESULTS[phase]

    cache_mode = f"{phase}_cache"

    dataset = create_dataset(
        cache_mode=cache_mode,
        split="train",
    )

    utils.print_section(f"{phase.upper()} HYPERPARAMETER TUNING")

    # OPTUNA STUDY

    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name=f"{phase}_hyperparameter_tuning"
    )

    # TOP LEVEL MLFLOW RUN

    with mlflow.start_run(run_name=f"{phase}_hyperparameter_tuning"):
        # GLOBAL SEARCH SETTINGS

        mlflow.log_params(
            {
                "phase": phase,
                "cache_mode": cache_mode,
                "n_trials": N_TRIALS,
                "n_splits": N_SPLITS,
                "batch_size": BATCH_SIZE,
                "val_batch_size": VAL_BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "seed": SEED,
                "device": str(DEVICE),
                "rating_max_error": RATING_MAX_ERROR,
                "embedding_dim": EMBEDDING_DIM,
            }
        )

        # PHASE 2 ARCHITECTURE CHANGES

        if phase == "phase2":
            mlflow.log_params(
                {
                    "trainable_vit_blocks": TRAINABLE_VIT_BLOCKS,
                    "trainable_bert_blocks": TRAINABLE_BERT_LAYERS,
                }
            )

        # SEARCH-SPACE DEFINITIONS

        # These describes what Optuna is allowed to search.

        mlflow.log_params(
            {
                "learning_rate_min": 1e-5,
                "learning_rate_max": 1e-4,
                "weight_decay_min": 1e-6,
                "weight_decay_max": 1e-2,
                "min_epochs": MIN_EPOCHS,
                "max_epochs": MAX_EPOCHS,
                "epoch_step": EPOCH_STEP,
                "tabular_dropout_min": MIN_DROPOUT,
                "tabular_dropout_max": MAX_DROPOUT,
                "tabular_dropout_step": DROPOUT_STEP,
                "attention_dropout_min": MIN_DROPOUT,
                "attention_dropout_max": MAX_DROPOUT,
                "attention_dropout_step": DROPOUT_STEP,
            }
        )

        if phase == "phase2":
            mlflow.log_params(
                {
                    "contrastive_temperature_min": MIN_CONTRASTIVE_TEMPERATURE,
                    "contrastive_temperature_max": MAX_CONTRASTIVE_TEMPERATURE,
                    "contrastive_temperature_scale": "log",
                }
            )

        # RUN OPTUNA

        study.optimize(
            lambda trial: objective(
                trial,
                dataset=dataset,
                phase=phase,
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

        save_json(best_config, best_config_path)

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

        save_json(cv_results, cv_results_path)

        mlflow.log_artifact(str(CV_RESULTS), artifact_path="metrics")

        # BEST-TRIAL SUMMARY ON TOP-LEVEL RUN

        mlflow.log_param("best_trial_number", best_trial.number)

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
    run_hyperparameter_tuning(phase="phase2")
