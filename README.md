# CineFusion-X

## Multimodal Movie Intelligence System

CineFusion-X is a multimodal, multitask movie intelligence system that combines:

- Movie posters
- Movie overview / plot text
- Structured movie metadata

The model learns separate visual, textual, and tabular representations, fuses them with cross-attention, and performs four movie-level tasks:

- **Genre** — multi-label classification
- **Rating** — regression
- **Revenue / Box Office** — regression
- **Certification / Content Rating** — multiclass classification

The training strategy has two phases:

```text
Phase 1 → Task-Only Training
        ↓
Cross-Validation + Hyperparameter Tuning
        ↓
Phase 2 → Multimodal Alignment + Joint Optimization
        ↓
Final Test Evaluation
```

---

# Project Pipeline

## 0. Data Preparation

These files prepare the data before any neural-network training begins.

```text
08_create_targets.py
        ↓
09_split_dataset.py
        ↓
preprocessing.py
        ↓
featurization.py
        ↓
custom_dataset.py
        ↓
data_loader.py
        ↓
validate_dataset.py
```

### `08_create_targets.py`

Creates the ground-truth targets for all four prediction tasks.

Responsibilities:

- Create the 19 genre multi-hot target columns.
- Create the rating regression target.
- Create the box-office / revenue target.
- Create the content-rating / certification target.
- Preserve missing target information for later task-specific masking.

---

### `09_split_dataset.py`

Creates the movie-level development/test split.

Responsibilities:

- Perform the chronological split.
- Use `imdb_id` as the movie identifier.
- Keep development and test movies separate.
- Apply the same movie-level split to the task-specific datasets.
- Validate that no movie appears in both splits.

The current chronological rule is:

```text
release_year < 2019  → Development
release_year >= 2019 → Test
```

---

### `preprocessing.py`

Handles preprocessing of **raw tabular model inputs only**.

Responsibilities:

- Numerical preprocessing
- Missing-value handling
- Categorical encoding
- Scaling
- Feature engineering
- Missingness indicators
- Cyclic date/time features

Targets are not treated as tabular model features.

---

### `featurization.py`

Applies the tabular preprocessor after the development/test split.

Responsibilities:

- Fit the tabular preprocessor on development data only.
- Transform development data.
- Transform test data with the same fitted preprocessor.
- Save processed tabular features.
- Add `imdb_id`, `overview`, and task targets back to the processed dataset.
- Save the fitted preprocessor.

---

### `custom_dataset.py`

Defines `MovieDataset`.

Responsibilities:

- Convert one movie into one PyTorch-ready sample.
- Load the corresponding poster.
- Apply the ViT image preprocessing.
- Retrieve/tokenize the movie overview for BERT.
- Retrieve processed tabular features.
- Retrieve all task targets.
- Retrieve target-availability masks.
- Return the sample in a format that can be batched by PyTorch.

It does **not** run ViT, BERT, the MLP, cross-attention, or task heads.

---

### `data_loader.py`

Creates the PyTorch `DataLoader`s.

Responsibilities:

- Create training/test Dataset objects.
- Group individual `MovieDataset` samples into batches.
- Provide batches to the training and evaluation pipelines.

---

### `validate_dataset.py`

Validates the complete data-to-model-input boundary.

Responsibilities:

- Validate target structure.
- Validate target encodings.
- Validate feature/target separation.
- Validate tensors, shapes, and dtypes.
- Validate target masks.
- Validate poster availability.
- Validate DataLoader batch creation.
- Validate that masked targets are handled correctly.

When this passes, the **data layer is complete**.

---

# 1. Model Components

These files are shared by both Phase 1 and Phase 2.

```text
vit_encoder.py
bert_encoder.py
tabular_encoder.py
        ↓
cross_attention.py
        ↓
task_heads.py
        ↓
cinefusion_model.py
```

### `vit_encoder.py`

Handles the visual branch.

```text
pixel_values
    ↓
ViT
    ↓
visual embedding
```

Responsibilities:

- Load the pretrained ViT.
- Process poster tensors.
- Produce visual representations.
- Fine-tune the ViT during training.

---

### `bert_encoder.py`

Handles the text branch.

```text
input_ids
attention_mask
        ↓
BERT
        ↓
text embedding
```

Responsibilities:

- Load pretrained BERT.
- Process tokenized movie overviews.
- Produce text representations.
- Fine-tune BERT during training.

---

### `tabular_encoder.py`

Handles the structured metadata branch.

```text
processed features
        ↓
MLP
        ↓
tabular embedding
```

Responsibilities:

- Receive processed tabular features.
- Map them into the model's learned representation space.

The MLP is trained from scratch.

---

### `cross_attention.py`

Handles multimodal fusion.

```text
Visual Embedding
Text Embedding
Tabular Embedding
        ↓
Cross-Attention
        ↓
Fused Representation
```

Responsibilities:

- Allow information from different modalities to interact.
- Produce the fused multimodal representation.

The cross-attention module is trainable.

---

### `task_heads.py`

Defines the four prediction heads.

```text
Fused Representation
        │
        ├── Genre Head
        ├── Rating Head
        ├── Revenue Head
        └── Certification Head
```

Responsibilities:

- Convert the fused representation into task-specific predictions.
- Produce the appropriate output dimensionality for each task.

It does **not** calculate losses.

---

### `cinefusion_model.py`

Defines the complete CineFusion-X model.

Responsibilities:

- Connect ViT, BERT, and tabular MLP.
- Run multimodal fusion through cross-attention.
- Pass the fused representation to the four task heads.
- Return the model predictions and/or representations required by the training phase.

This is the main model-level forward-pass module.

---

# 2. Phase 1 — Task-Only Training

## Objective

Phase 1 first teaches the entire multimodal model to solve the four supervised tasks.

```text
Batch
  ↓
ViT + BERT + MLP
  ↓
Cross-Attention
  ↓
Fused Representation
  ↓
Task Heads
  ↓
Four Predictions
  ↓
Task Losses
  ↓
Backpropagation
  ↓
Parameter Update
```

The Phase-1 objective is:

```text
L_phase1 = L_task
```

---

### `model_losses.py` 

Handles the supervised multitask objective.

Responsibilities:

- Calculate genre loss.
- Calculate rating loss.
- Calculate revenue / box-office loss.
- Calculate certification / content-rating loss.
- Apply task-specific target masks.
- Combine the per-task losses into the total supervised loss.

Conceptually:

```text
L_task =
    λ_genre L_genre
  + λ_rating L_rating
  + λ_revenue L_revenue
  + λ_cert L_cert
```

A missing target is ignored for its specific task instead of removing the entire movie from the batch.

---

### `train_phase_1.py`

train_phase_1.py is responsible for executing one actual Phase 1 training run on one train/validation split. It receives the DataLoaders and hyperparameters, creates the CineFusionModel, MultiTaskLoss, and AdamW optimizer, then iterates through the requested number of epochs. For every epoch it trains batch-by-batch using the multitask masked loss, then switches to evaluation mode and processes the validation set. The validation outputs from all batches are collected and concatenated so that Genre, Rating, Box Office, and Content Rating metrics are calculated on the complete validation fold. These task metrics are then combined into the composite score, and the epoch with the highest composite score becomes the best epoch. When a checkpoint path is supplied, the model and optimizer states plus relevant training metadata are saved through torch.save().

From the MLflow perspective, this file is responsible for training-related metrics and the checkpoint artifact, not for defining the overall Optuna search. In CV mode (return_full_metrics=False), it keeps the run lightweight and logs the best validation loss, best epoch, and best composite score once for that fold. In detailed best-configuration mode (return_full_metrics=True), it additionally logs epoch-by-epoch train loss, validation loss, and composite score, followed by overall task metrics and all per-genre/per-class metrics. If a checkpoint path is provided, the saved checkpoint is also registered as an MLflow artifact. The function finally returns only the best composite score in CV mode, while returning the complete best_metrics dictionary in detailed mode.

### `cross_validation.py`

cross_validation.py is responsible for evaluating one hyperparameter configuration across multiple time-ordered folds. It creates a TimeSeriesSplit, generates the training and validation indices for each fold, builds Subset objects and their DataLoaders, and then calls train_phase_1() once for each fold with return_full_metrics=False. Each fold therefore trains its own model and returns only its best composite score to the cross-validation layer. The five fold scores are collected and then aggregated into the mean and standard deviation of the composite score. These aggregated values represent how well that one configuration performs across the different time-ordered validation splits, and the mean composite score is the value that the hyperparameter-search layer uses for comparison. Test data is never involved in this process.

MLflow here is used at the fold level and configuration/CV-summary level. Each fold is created as a nested MLflow run, where the fold number, number of splits, seed, batch size, device, and the hyperparameter configuration being evaluated are recorded. The actual best-fold training metrics such as best_val_loss, best_epoch, and best_composite_score are produced by train_phase_1(). After all five folds finish, cross_validation.py aggregates their composite scores and logs the configuration-level mean_cv_composite_score and std_cv_composite_score. Thus, this file provides the MLflow hierarchy that lets you inspect individual folds while still having a single aggregated result for the configuration.

### `hyperp_tuning.py`

hyperp_tuning.py is the top-level Phase 1 search and orchestration layer. It creates the Optuna study, defines the search space through trial.suggest_float, trial.suggest_int, and trial.suggest_categorical, and repeatedly calls the cross-validation pipeline for each trial. Each trial therefore represents one candidate hyperparameter configuration, and its objective is the mean composite score returned by cross-validation. Optuna compares these trial results and selects the configuration with the highest mean CV composite score. Once the winning configuration is identified, this file saves phase1_best_config.json, creates the final Phase 1 train/validation split for the detailed run, calls train_phase_1() again with return_full_metrics=True, and saves the returned complete best_metrics as phase1_best_metrics.json. It also preserves the overall Optuna/CV search history in phase1_cv_results.json.

MLflow in hyperp_tuning.py operates at the highest level, representing the Optuna trial and the final selected configuration. A parent trial run records the candidate hyperparameters, while the nested fold runs created by cross_validation.py contain the detailed fold-level information. The trial ultimately receives the aggregated mean CV composite score that determines which configuration wins. After Optuna chooses the winner, a separate nested phase1_best_configuration run records the selected configuration and the detailed training history and metrics generated by train_phase_1(). The final JSON artifacts are saved by the filesystem side of this orchestration, while MLflow tracks the experiment history and the relevant artifacts. The checkpoint itself is owned by train_phase_1.py; hyperp_tuning.py should not upload that same checkpoint a second time.

# 4. Phase 2 — Multimodal Alignment + Joint Optimization

## Objective

Phase 2 starts from the **best Phase-1 checkpoint** and the configuration selected through development-set cross-validation and hyperparameter tuning.

Phase 2 adds an explicit multimodal contrastive objective while continuing to optimize the supervised task objective.

```text
Best Phase-1 Checkpoint
          ↓
Current Batch
          ↓
ViT + BERT + MLP
          ↓
Modality Embeddings
          ↓
Contrastive Alignment
          +
Cross-Attention Fusion
          ↓
Fused Representation
          ↓
Task Heads
          ↓
Task Predictions
          ↓
Task Losses
          +
Contrastive Loss
          ↓
Total Phase-2 Loss
          ↓
Backpropagation
          ↓
Parameter Update
```

The Phase-2 objective is:

```text
L_phase2 =
    L_task
  + λ_contrastive L_contrastive
```

The same model architecture is reused from Phase 1.

There is **no second copy of ViT, BERT, MLP, Cross-Attention, or the Task Heads**.

---

### `contrastive_loss.py`

Contains only the contrastive learning objective.

Responsibilities:

- Receive modality embeddings.
- Construct positive same-movie relationships.
- Use other movies as negatives according to the contrastive formulation.
- Calculate the contrastive loss.

Conceptually:

```text
Same movie
   ↓
Visual ↔ Text ↔ Tabular
   ↓
positive relationships

Different movies
   ↓
negative relationships
```

The goal is to improve alignment between the representations of different modalities belonging to the same movie.

---

### `train_phase_2.py`

Runs Phase-2 training.

Responsibilities:

- Load the best Phase-1 checkpoint.
- Reuse the existing CineFusion-X model.
- Generate modality embeddings.
- Calculate the contrastive loss.
- Run cross-attention fusion.
- Generate four task predictions.
- Calculate the existing supervised task losses.
- Combine the objectives:

```text
L_total =
    L_task
  + λ_contrastive L_contrastive
```

- Backpropagate through all trainable components.
- Update the model parameters.
- Track development performance.
- Save the best final checkpoint.

Phase 2 therefore performs **joint optimization** of:

```text
Supervised Task Learning
+
Multimodal Representation Alignment
```

---

### Phase-2 Checkpoint

The best Phase-2 checkpoint becomes the **best final checkpoint**.

It contains the jointly optimized parameters of:

```text
ViT
BERT
Tabular MLP
Cross-Attention
Task Heads
```

The model has now been trained using both:

```text
Task objective
+
Contrastive alignment objective
```

---

# 5. Final Evaluation

## `evaluate.py`

The final evaluation happens only after Phase 2 is complete.

```text
Best Final Checkpoint
        ↓
evaluate.py
        ↓
Untouched Test Data
        ↓
Forward Pass
        ↓
Four Predictions
        ↓
metrics.py
        ↓
Final Test Metrics
        ↓
Report Results
```

Responsibilities:

- Load the best final checkpoint.
- Run the final model on the untouched test set.
- Generate predictions for all four tasks.
- Use `metrics.py` to calculate the final reported metrics.
- Report the final results.

The test set is **not** used for:

- training
- cross-validation
- hyperparameter tuning
- Phase-2 checkpoint selection

It is used only for the final evaluation.

---

# Complete File Pipeline

## Data Layer

```text
08_create_targets.py
        ↓
09_split_dataset.py
        ↓
preprocessing.py
        ↓
featurization.py
        ↓
custom_dataset.py
        ↓
data_loader.py
        ↓
validate_dataset.py
        ↓
DATA LAYER COMPLETE
```

## Shared Model Layer

These files are created once and reused in both training phases.

```text
vit_encoder.py
bert_encoder.py
tabular_encoder.py
        ↓
cross_attention.py
        ↓
task_heads.py
        ↓
cinefusion_model.py
```

## Phase 1 — Task-Only Training

```text
model_losses.py
        ↓
train_phase_1.py
        ↓
Best Phase-1 Checkpoint
```

Phase 1 uses the shared model components above and the existing supervised
loss implementation. No second copy of the model is created.

## Phase 2 Model Selection

After Phase 1 produces a working/best checkpoint, Phase-2 hyperparameters
are selected using only the development data.

```text
Best Phase-1 Checkpoint
        ↓
hyperparameter_tuning.py
        ↕
cross_validation.py
        ↓
metrics.py
        ↓
Best Phase-2 Configuration
```

`metrics.py` is a shared utility. It is used by cross-validation/model
selection and later by final evaluation; it is not a separate training
stage.

## Phase 2 — Multimodal Alignment + Joint Optimization

```text
Best Phase-1 Checkpoint
        ↓
train_phase_2.py
        │
        ├── reuses vit_encoder.py
        ├── reuses bert_encoder.py
        ├── reuses tabular_encoder.py
        ├── reuses cross_attention.py
        ├── reuses task_heads.py
        ├── reuses cinefusion_model.py
        └── reuses model_losses.py
                  +
            contrastive_loss.py
        ↓
Best Final Checkpoint
```

Phase 2 does not duplicate the Phase-1 architecture. It continues from the
Phase-1 checkpoint and adds the contrastive objective to the existing
supervised objective.

## Final Evaluation

```text
Best Final Checkpoint
        ↓
evaluate.py
        ↓
metrics.py
        ↓
Untouched Test Set
        ↓
Final Metrics
```

# Final End-to-End Flow

```text
DATA
  ↓
08_create_targets.py
  ↓
09_split_dataset.py
  ↓
preprocessing.py
  ↓
featurization.py
  ↓
custom_dataset.py
  ↓
data_loader.py
  ↓
validate_dataset.py
  ↓
DATA LAYER COMPLETE
  ↓
────────────────────────────────────────
SHARED MODEL
────────────────────────────────────────
vit_encoder.py
bert_encoder.py
tabular_encoder.py
  ↓
cross_attention.py
  ↓
task_heads.py
  ↓
cinefusion_model.py
  ↓
────────────────────────────────────────
PHASE 1 — TASK-ONLY TRAINING
────────────────────────────────────────
model_losses.py
  ↓
train_phase_1.py
  ↓
Best Phase-1 Checkpoint
  ↓
────────────────────────────────────────
PHASE 2 MODEL SELECTION
────────────────────────────────────────
hyperparameter_tuning.py
        ↕
cross_validation.py
        ↓
metrics.py
        ↓
Best Phase-2 Configuration
  ↓
────────────────────────────────────────
PHASE 2 — MULTIMODAL ALIGNMENT
       + JOINT OPTIMIZATION
────────────────────────────────────────
Best Phase-1 Checkpoint
  ↓
contrastive_loss.py
        +
existing model_losses.py
  ↓
train_phase_2.py
  ↓
Best Final Checkpoint
  ↓
────────────────────────────────────────
FINAL EVALUATION
────────────────────────────────────────
evaluate.py
  ↓
metrics.py
  ↓
UNTOUCHED TEST SET
  ↓
FINAL METRICS
```

# Design Principles

### Reuse the same model

Phase 2 does not duplicate the architecture from Phase 1.

The same:

```text
ViT
BERT
Tabular MLP
Cross-Attention
Task Heads
```

are reused.

### Reuse the same supervised loss

`model_losses.py` remains responsible for the four task losses in both phases.

Phase 2 only adds:

```text
contrastive_loss.py
```

to the existing supervised objective.

### Keep model selection away from the test set

Cross-validation and hyperparameter tuning operate on development data.

The test set is reserved for the final evaluation.

### Keep metric calculation centralized

`metrics.py` provides the common metric implementation used by:

- cross-validation
- final evaluation

---

# Current Development Status

```text
Data preparation                 ✓
Target creation                  ✓
Chronological splitting          ✓
Tabular preprocessing            ✓
Featurization                    ✓
MovieDataset                     ✓
DataLoader                       ✓
Dataset validation               ✓

Next:
ViT encoder
BERT encoder
Tabular encoder
Cross-Attention
Task Heads
CineFusion-X model
Model losses
Metrics
Phase 1 training
Cross-validation
Hyperparameter tuning
Contrastive loss
Phase 2 training
Final evaluation
```
