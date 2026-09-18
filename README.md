# CineFusion-X

## Multimodal Movie Intelligence System

CineFusion-X is a multimodal, multitask movie intelligence system that combines:

- **Movie posters** → visual information
- **Movie overview / plot text** → textual information
- **Structured movie metadata** → tabular information

The system learns a representation from each modality, aligns the modalities during Phase 2, fuses them through multimodal attention, and predicts four movie-level tasks:

| Task | Type | Output |
|---|---|---|
| Genre | Multi-label classification | 19 independent genre probabilities |
| Rating | Regression | Continuous rating prediction |
| Box Office | 4-class classification | Flop / Average / Hit / Blockbuster |
| Content Rating | 4-class classification | G / PG / PG-13 / R |

The training and evaluation strategy is:

```text
Development Data
      ↓
Phase 1: Task-Only Training
      ↓
TimeSeries Cross-Validation + Optuna
      ↓
Best Phase-1 Configuration
      ↓
Final Phase-1 Training on ALL Development Data
      ↓
Best Phase-1 Checkpoint
      ↓
Phase 2: Alignment + Joint Optimization
      ↓
Final Phase-2 Model
      ↓
Development Calibration
      ↓
Temperature Scaling + Genre Threshold Finding
      ↓
Locked Calibration Parameters
      ↓
Untouched 2019+ Test Set
      ↓
Final Metrics + Optional Uncertainty
```

The **test set is never used for training, hyperparameter selection, checkpoint selection, threshold finding, calibration, or early stopping**.

---

# 1. Data Split and Leakage Policy

The project uses a chronological movie-level split:

```text
release_year < 2019  → Development
release_year >= 2019 → Test
```

The split is performed by movie identifier (`imdb_id`) so that the same movie cannot appear in both partitions.

```text
                    ALL MOVIES
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
     DEVELOPMENT                 TEST
     < 2019                      >= 2019
            │                       │
     CV / Optuna /            FINAL EVALUATION
     Phase 1 / Phase 2
     / calibration
```

## Leakage rules

The test set must not be used for:

- model training
- cross-validation
- Optuna trials
- hyperparameter selection
- checkpoint selection
- early stopping
- threshold finding
- temperature scaling
- uncertainty configuration
- preprocessing fitting
- feature-scaling fitting
- category mapping creation
- decision-rule tuning

The test set is used only after the model-selection and calibration pipeline is locked.

---

# 2. Project Structure

```text
src/
├── dataset/
│   ├── 08_create_targets.py
│   ├── 09_split_dataset.py
│   ├── preprocessing.py
│   ├── featurization.py
│   ├── custom_dataset.py
│   ├── data_loader.py
│   └── validate_dataset.py
│
├── models/
│   ├── vit_encoder.py
│   ├── bert_encoder.py
│   ├── tabular_encoder.py
│   ├── cross_attention.py
│   ├── task_heads.py
│   ├── cinefusion_model.py
│   └── contrastive_loss.py
│
├── losses/
│   ├── model_losses.py
│   └── phase2_losses.py
│
├── training/
│   ├── train_phase_1.py
│   ├── cross_validation.py
│   ├── hyperp_tuning.py
│   ├── train_final.py
│   └── train_phase_2.py
│
├── evaluation/
│   ├── metrics.py
│   ├── calibration.py
│   └── uncertainty.py
│
└── utils/
    └── embedding_cache.py
```

Each file has one clear responsibility. Training orchestration, loss definitions, metric calculation, calibration, and uncertainty estimation are kept separate so that the same model implementation can be reused without creating duplicate model code.

---

# 3. Data Layer

The data layer converts raw movie information into model-ready samples.

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

## `08_create_targets.py`

Creates the ground-truth targets used by the four supervised tasks.

### Responsibilities

- Create the 19 genre multi-hot target columns.
- Create the movie-rating regression target.
- Create the box-office target and its four class labels.
- Create the content-rating / certification target.
- Preserve missing-target information so individual tasks can ignore unavailable targets.

The resulting task structure is:

```text
Genre
→ 19 binary labels

Rating
→ continuous value

Box Office
→ 0 = Flop
→ 1 = Average
→ 2 = Hit
→ 3 = Blockbuster

Content Rating
→ 0 = G
→ 1 = PG
→ 2 = PG-13
→ 3 = R
```

---

## `09_split_dataset.py`

Creates the chronological development/test split.

### Responsibilities

- Split movies chronologically.
- Use `imdb_id` as the movie identifier.
- Keep development and test movies separate.
- Apply the same movie-level split to task-specific datasets.
- Validate that no movie appears in both partitions.

Current split:

```text
release_year < 2019  → Development
release_year >= 2019 → Test
```

---

## `preprocessing.py`

Handles preprocessing of raw **tabular model inputs**.

### Responsibilities

- Numerical preprocessing.
- Missing-value handling.
- Categorical encoding.
- Scaling.
- Feature engineering.
- Missingness indicators.
- Cyclic date/time features.

Targets are not treated as ordinary tabular input features.

---

## `featurization.py`

Fits and applies the tabular preprocessing pipeline after the development/test split.

### Responsibilities

- Fit the tabular preprocessor on development data only.
- Transform development data.
- Transform test data using the already-fitted development preprocessor.
- Save processed tabular features.
- Preserve `imdb_id`, overview text, and task targets.
- Save the fitted preprocessing object.

This prevents future test information from affecting feature preprocessing.

---

## `custom_dataset.py`

Defines `MovieDataset`.

For each movie, it produces the tensors consumed by the model.

```text
Movie
 │
 ├── Poster
 │     └── ViT image preprocessing
 │
 ├── Overview
 │     └── BERT tokenization
 │
 ├── Tabular features
 │
 └── Targets + target masks
```

### Responsibilities

- Load the corresponding poster.
- Apply image preprocessing.
- Tokenize the movie overview.
- Retrieve processed tabular features.
- Retrieve all task targets.
- Retrieve task-specific target-availability masks.
- Return a PyTorch-ready sample.

`custom_dataset.py` does **not** run ViT, BERT, the MLP, cross-attention, or task heads.

---

## `data_loader.py`

Creates PyTorch `DataLoader`s.

### Responsibilities

- Construct development/test Dataset objects.
- Batch `MovieDataset` samples.
- Configure batching and worker behavior.
- Provide batches to training and evaluation pipelines.

---

## `validate_dataset.py`

Validates the complete boundary between data preparation and model input.

### Responsibilities

- Validate target structure.
- Validate target encodings.
- Validate feature/target separation.
- Validate tensor shapes and dtypes.
- Validate target masks.
- Validate poster availability.
- Validate DataLoader batch creation.
- Validate that missing targets are masked correctly.

When this passes, the data layer is ready for model training.

---

# 4. Model Layer

The architecture is defined once and reused by both training phases.

```text
Poster ──────→ ViT Encoder ────────→ Visual Embedding
Overview ────→ BERT Encoder ───────→ Text Embedding
Metadata ────→ Tabular MLP ────────→ Tabular Embedding
                                          │
                                          ▼
                                 Multimodal Attention
                                          │
                                          ▼
                                  Fused Representation
                                          │
                                          ▼
                                      Task Heads
```

There is only one implementation of each encoder, fusion module, task-head module, and complete CineFusion-X model.

---

## `vit_encoder.py`

The visual encoder.

```text
Poster
  ↓
Image preprocessing
  ↓
MAE-pretrained ViT-B/16
  ↓
Visual feature vector
  ↓
Projection
  ↓
Visual embedding
```

Current visual backbone:

```text
vit_base_patch16_224.mae
```

### Responsibilities

- Load the pretrained ViT.
- Process poster tensors.
- Produce visual representations.
- Project the backbone representation into the shared embedding dimension.
- Support selective fine-tuning of the final ViT blocks for Phase 2.

### Phase 1

```text
ViT backbone  → frozen
ViT projection → trainable
```

### Phase 2

The final `N` transformer blocks can be selectively unfrozen while earlier blocks remain frozen.

---

## `bert_encoder.py`

The textual encoder.

```text
Overview tokens
      ↓
BERT-base-uncased
      ↓
[CLS] representation
      ↓
Projection
      ↓
Text embedding
```

BERT-base has a 768-dimensional hidden representation before projection.

### Responsibilities

- Load pretrained BERT.
- Process tokenized movie overviews.
- Produce the `[CLS]` representation.
- Project it into the shared embedding dimension.
- Support selective fine-tuning of the final BERT layers for Phase 2.

### Phase 1

```text
BERT backbone   → frozen
BERT projection → trainable
```

### Phase 2

The final `N` BERT transformer layers can be selectively unfrozen while earlier layers remain frozen.

---

## `tabular_encoder.py`

Handles structured movie metadata.

```text
Processed tabular features
          ↓
         MLP
          ↓
   Tabular embedding
```

The MLP is trained from scratch.

---

## `cross_attention.py`

Performs multimodal fusion.

```text
Visual Embedding
Text Embedding ───→ Multimodal Attention ───→ Fused Representation
Tabular Embedding
```

### Responsibilities

- Allow information from different modalities to interact.
- Combine visual, textual, and tabular information.
- Produce the fused representation used by the task heads.

The fusion module is trainable.

---

## `task_heads.py`

Defines the four supervised prediction heads.

```text
                  Fused Representation
                         │
        ┌────────────────┼─────────────────┐
        │                │                 │
        ▼                ▼                 ▼
     Genre            Rating           Box Office
   19 logits        regression       4-class logits
                         │
                         ▼
                 Content Rating
                   4-class logits
```

### Responsibilities

- Convert the fused representation into task-specific outputs.
- Produce the correct output dimensionality for each task.

It does not calculate losses or evaluation metrics.

---

## `cinefusion_model.py`

Connects the complete architecture.

### Forward path

```text
pixel_values
     ↓
ViT
     ↓
visual embedding

input_ids + attention_mask
     ↓
BERT
     ↓
text embedding

tabular features
     ↓
MLP
     ↓
tabular embedding

visual + text + tabular
     ↓
multimodal attention
     ↓
fused representation
     ↓
task heads
     ↓
four task outputs
```

### Responsibilities

- Instantiate the three modality encoders.
- Instantiate multimodal attention.
- Instantiate the task heads.
- Execute the complete forward pass.
- Return modality embeddings, fused representation, and task predictions required by training and evaluation.

Phase 2 reuses this same model rather than creating a second architecture.

---

## `contrastive_loss.py`

Contains the modality-alignment objective used in Phase 2.

### Responsibilities

- Receive modality embeddings.
- Construct positive relationships between modalities belonging to the same movie.
- Use other movies as negatives according to the chosen contrastive formulation.
- Calculate the contrastive loss.

Conceptually:

```text
Same movie
   ├── Visual ↔ Text
   ├── Visual ↔ Tabular
   └── Text   ↔ Tabular
           ↓
       Positives

Different movies
           ↓
        Negatives
```

---

# 5. Loss Layer

## `model_losses.py`

Contains the supervised multitask loss used in both Phase 1 and Phase 2.

```text
L_task =
    λ_genre L_genre
  + λ_rating L_rating
  + λ_box_office L_box_office
  + λ_content_rating L_content_rating
```

### Task losses

```text
Genre
→ BCEWithLogitsLoss

Rating
→ SmoothL1Loss

Box Office
→ CrossEntropyLoss

Content Rating
→ CrossEntropyLoss
```

Target masks are applied independently to each task. A missing target for one task does not remove that movie from the other task losses.

Content-rating class weights can be supplied to the classification loss to address class imbalance.

---

## `phase2_losses.py`

Contains Phase-2-specific loss composition when additional Phase-2 objective handling is needed.

The core Phase-2 objective remains:

```text
L_phase2 =
    L_task
  + λ_contrastive L_contrastive
```

`model_losses.py` remains the source of truth for the four supervised task losses.

`contrastive_loss.py` remains the implementation of the contrastive objective.

`phase2_losses.py` composes them where Phase-2-specific handling is required; it does not duplicate the four task-loss implementations.

---

# 6. Phase 1 — Task-Only Training

## Objective

Phase 1 teaches the multimodal model to solve the four supervised tasks before explicit representation alignment is introduced.

```text
Batch
  ↓
ViT + BERT + Tabular MLP
  ↓
Visual / Text / Tabular Embeddings
  ↓
Multimodal Attention
  ↓
Fused Representation
  ↓
Task Heads
  ↓
Four Predictions
  ↓
Masked Task Losses
  ↓
Backpropagation
  ↓
Parameter Update
```

```text
L_phase1 = L_task
```

### Trainable components in Phase 1

```text
ViT backbone        → frozen
BERT backbone       → frozen
ViT projection      → trainable
BERT projection     → trainable
Tabular MLP         → trainable
Cross-Attention     → trainable
Task Heads          → trainable
```

---

## `train_phase_1.py`

The single-configuration training engine used by cross-validation.

### Responsibilities

- Create the CineFusion-X model from the supplied configuration.
- Create `MultiTaskLoss`.
- Create the optimizer and validation-driven scheduler.
- Train one configuration on one development train/validation split.
- Evaluate the validation split after every epoch.
- Calculate task metrics on the complete validation fold.
- Calculate the composite development score.
- Track the best validation epoch.
- Optionally save the best checkpoint when a checkpoint path is supplied.
- Return the best composite score or the full best-metrics object depending on the caller.

This file does **not** run Optuna and does **not** perform final test evaluation.

---

# 7. Cross-Validation

## `cross_validation.py`

Performs chronological cross-validation over the development set only.

```text
Development Data
      ↓
TimeSeriesSplit
      ↓
Fold 1 ──┐
Fold 2 ──┼──→ train_phase_1.py
Fold 3 ──┘
      ↓
Best score from each fold
      ↓
Mean + Standard Deviation
```

### Responsibilities

- Create time-ordered development folds.
- Create train/validation `Subset`s and DataLoaders.
- Call `train_phase_1.py` once per fold.
- Collect each fold's best composite score.
- Calculate mean and standard deviation.
- Return the aggregated result to the hyperparameter-search layer.

### It does not

- access the actual test set
- perform final training
- perform final test evaluation
- learn test-set thresholds
- calibrate on the test set
- create the final production checkpoint

---

# 8. Hyperparameter Tuning

## `hyperp_tuning.py`

Top-level Phase-1 model-selection orchestrator.

```text
Optuna Trial
     ↓
Candidate Configuration
     ↓
cross_validation.py
     ↓
TimeSeriesSplit
     ↓
train_phase_1.py
     ↓
Fold Scores
     ↓
Mean CV Composite Score
     ↓
Optuna
     ↓
Best Configuration
```

### Responsibilities

- Create the Optuna study.
- Define the Phase-1 search space.
- Run trials.
- Call cross-validation for each candidate configuration.
- Optimize mean development CV composite score.
- Record fold variability.
- Save the selected configuration and CV history.

Typical search parameters can include learning rate, training budget, tabular hidden dimension, embedding dimension, task weights, weight decay, and relevant regularization settings.

### Important distinction

```text
Best configuration
≠
Best learned weights
```

Optuna selects the configuration. Final training produces the final learned weights.

Typical outputs:

```text
phase1_best_config.json
phase1_cv_results.json
```

---

# 9. Final Training

## `train_final.py`

Final-training orchestrator used after development-set model selection.

The final model is **not** trained only on the last CV fold.

```text
Development Data
      ↓
Optuna + Cross-Validation
      ↓
Best Configuration
      ↓
ALL Development Data
      ↓
Final Training
      ↓
Final Checkpoint
```

### Responsibilities

- Load the selected configuration.
- Build the final training Dataset/DataLoader over all development data.
- Train the selected model using a fixed final training budget derived from development-stage results.
- Save the final checkpoint.
- Keep the actual test set completely untouched until final evaluation.

Because the final run uses all development data, it should not depend on test-set performance for early stopping or checkpoint selection.

`train_final.py` is an orchestration layer rather than a second copy of the full training loop.

---

# 10. Phase 2 — Multimodal Alignment + Joint Optimization

## Objective

Phase 2 starts from the best Phase-1 model and adds an explicit multimodal alignment objective.

```text
Best Phase-1 Checkpoint
        ↓
Current Batch
        ↓
ViT + BERT + Tabular MLP
        ↓
Modality Embeddings
        ├──────────────→ Contrastive Loss
        │
        ↓
Multimodal Attention
        ↓
Fused Representation
        ↓
Task Heads
        ↓
Supervised Task Loss
        │
        └──────────────┐
                       ▼
               Total Phase-2 Loss
                       ↓
                 Backpropagation
                       ↓
                 Parameter Update
```

```text
L_phase2 =
    L_task
  + λ_contrastive L_contrastive
```

Phase 2 optimizes:

```text
Supervised Task Learning
+
Multimodal Representation Alignment
```

---

## Selective backbone fine-tuning

Phase 2 can selectively fine-tune only the final transformer blocks/layers:

```text
ViT:
    frozen earlier blocks
            +
    trainable final N blocks

BERT:
    frozen earlier layers
            +
    trainable final N layers
```

The number of trainable blocks/layers is a Phase-2 configuration parameter and can be evaluated through development-only experiments/ablations.

---

## `train_phase_2.py`

Phase-2 training engine.

### Responsibilities

- Load the best Phase-1 checkpoint.
- Configure selective ViT/BERT fine-tuning.
- Reuse the existing tabular encoder.
- Generate modality embeddings.
- Calculate the contrastive loss.
- Run multimodal fusion.
- Produce the four task predictions.
- Calculate the existing supervised task losses.
- Combine supervised and contrastive losses.
- Train the jointly optimized model.
- Track development validation performance when running Phase-2 experiments.
- Produce the Phase-2 final checkpoint after the Phase-2 configuration is selected.

Phase 2 does not create a second copy of the model architecture.

---

# 11. Embedding Caching

## `embedding_cache.py`

Optional utility for avoiding repeated computation through frozen backbone sections.

### Phase 1

Because the ViT and BERT backbones are frozen, their backbone outputs can be cached before the trainable projection layers.

```text
Poster
  ↓
Frozen ViT backbone
  ↓
Cached backbone representation
  ↓
Trainable projection
  ↓
Visual embedding
```

```text
Overview
  ↓
Frozen BERT backbone
  ↓
Cached 768-d representation
  ↓
Trainable projection
  ↓
Text embedding
```

The projected 256-dimensional output should not be cached while the projection remains trainable.

### Phase 2

When only the final `N` transformer layers/blocks are trainable, the frozen prefix can be cached:

```text
Input
  ↓
Frozen prefix
  ↓
Cached intermediate representation
  ↓
Trainable final N layers/blocks
  ↓
Projection
```

The exact cache boundary must follow the real backbone forward path.

`torch.no_grad()` is not the same as caching:

```text
torch.no_grad()
→ avoids autograd graph construction

embedding_cache.py
→ stores reusable representations
```

---

# 12. Metrics

## `metrics.py`

`metrics.py` is the centralized evaluation utility.

Its responsibility is:

> Measure model performance.

It does not learn thresholds or calibration parameters.

---

## Genre metrics

Genre is a 19-label multilabel problem.

The model produces 19 sigmoid probabilities:

```text
[N, 19]
```

Discrete predictions require thresholds:

```text
probability >= threshold
        ↓
prediction = 1
```

Phase 1 uses:

```text
threshold = 0.5
```

Phase 2 can use thresholds learned by `calibration.py`.

Reported metrics can include:

- macro F1
- micro F1
- macro precision
- macro recall
- ROC-AUC
- per-genre F1
- per-genre precision
- per-genre recall
- per-genre ROC-AUC

---

## ROC-AUC

ROC-AUC is an evaluation metric and does **not** require one operating threshold.

Therefore:

```text
calibration.py
→ learns thresholds

metrics.py
→ calculates ROC-AUC and threshold-based metrics
```

For the 4-class classification tasks, multiclass ROC-AUC can be calculated from the full probability matrix using a one-vs-rest formulation.

---

## Box-office metrics

Box office is a 4-class classification task:

```text
0 = Flop
1 = Average
2 = Hit
3 = Blockbuster
```

Prediction uses the highest class probability:

```text
predicted_class = argmax(class_probabilities)
```

Metrics can include macro/weighted F1, accuracy, balanced accuracy, macro precision/recall, multiclass ROC-AUC, per-class metrics, and a confusion matrix.

---

## Content-rating metrics

Content rating is a 4-class classification task:

```text
0 = G
1 = PG
2 = PG-13
3 = R
```

Prediction uses `argmax` over the four class probabilities.

Metrics can include macro/weighted F1, accuracy, balanced accuracy, macro precision/recall, multiclass ROC-AUC, per-class metrics, and a confusion matrix.

---

## Rating metrics

Rating is regression.

Metrics:

- MAE
- RMSE

A normalized rating score is derived from MAE for the composite development score.

---

## Composite development score

The existing development-stage model-selection score is:

```text
rating_score =
max(
    0,
    1 - rating_mae / rating_max_error
)

composite_score =
(
    genre_macro_f1
    + rating_score
    + box_office_macro_f1
    + content_rating_macro_f1
) / 4
```

This composite score is the model-selection objective used by the current Phase-1 CV/Optuna pipeline.

ROC-AUC is an additional evaluation metric; it does not automatically replace the existing composite objective.

---

# 13. Calibration

## `calibration.py`

`calibration.py` handles post-training calibration and decision-rule tuning using development calibration data.

It contains two main jobs:

```text
calibration.py
├── Temperature Scaling
└── Threshold Finding
```

---

## Temperature scaling

Temperature scaling adjusts the confidence of classification outputs.

Conceptually:

```text
Raw model logits
      ↓
Temperature scaling
      ↓
Calibrated probabilities
```

The temperature parameter is learned using a calibration/validation portion of the development data.

It is locked before test evaluation.

---

## Threshold finding

Threshold finding is mainly relevant to the 19-label genre task.

The calibration data contains:

```text
Predicted probabilities
+
Ground-truth genre labels
```

The calibration procedure searches for the chosen operating threshold rule and saves the resulting thresholds.

A per-genre threshold representation is supported conceptually:

```text
genre_thresholds
{
    genre_0: ...,
    genre_1: ...,
    ...
    genre_18: ...
}
```

The division of responsibility is strict:

```text
calibration.py
    ↓
LEARN thresholds

metrics.py
    ↓
USE thresholds
and calculate metrics
```

Threshold finding is never performed on the 2019+ test labels.

---

# 14. Uncertainty Estimation

## `uncertainty.py`

Provides post-training uncertainty estimation using Monte Carlo Dropout.

Conceptually:

```text
Trained model
     ↓
Enable dropout during inference
     ↓
Forward pass 1
Forward pass 2
Forward pass 3
...
Forward pass K
     ↓
Aggregate predictions
     ↓
Mean + uncertainty estimate
```

This is an evaluation/inference utility, not another training loss.

The implementation must explicitly control which dropout modules are active during MC inference.

---

# 15. Final Evaluation

## `evaluate.py`

`evaluate.py` is the final inference/evaluation orchestrator.

### Responsibilities

- Load the final checkpoint to be evaluated.
- Load locked calibration parameters.
- Run inference on the untouched test set.
- Apply temperature scaling where configured.
- Apply locked genre thresholds.
- Collect predictions and targets.
- Calculate final metrics through `metrics.py`.
- Optionally calculate MC-Dropout uncertainty through `uncertainty.py`.
- Save final evaluation results.

The final evaluation order is:

```text
Final model
    ↓
Development calibration data
    ↓
calibration.py
    ├── temperature
    └── thresholds
    ↓
LOCK calibration parameters
    ↓
2019+ test set
    ↓
Forward pass
    ↓
Apply calibration
    ↓
Optional uncertainty
    ↓
metrics.py
    ↓
Final reported metrics
```

---

# 16. Phase 1 vs Phase 2 Comparison

Both phases must be evaluated under the same final test protocol.

```text
                    DEVELOPMENT
                         │
          ┌──────────────┴──────────────┐
          │                             │
       PHASE 1                       PHASE 2
          │                             │
   model selection                config/ablation
          │                             │
   final train on                final train on
   ALL development               ALL development
          │                             │
          ▼                             ▼
   Phase-1 final                  Phase-2 final
      model                          model
          │                             │
          └──────────────┬──────────────┘
                         ▼
                SAME 2019+ TEST SET
                         │
                         ▼
                SAME CALIBRATION RULES
                         │
                         ▼
                SAME METRIC DEFINITIONS
                         │
                         ▼
                 Comparable Results
```

Neither model gets an advantage from using a different test set, different metric definitions, or test-specific threshold tuning.

---

# 17. Complete File Responsibility Map

| File | Primary responsibility | Test data used? |
|---|---|---:|
| `08_create_targets.py` | Create supervised targets | No |
| `09_split_dataset.py` | Chronological development/test split | Only to define the partition |
| `preprocessing.py` | Raw tabular preprocessing logic | No test fitting |
| `featurization.py` | Fit development preprocessing and transform partitions | No test fitting |
| `custom_dataset.py` | Build PyTorch-ready movie samples | No |
| `data_loader.py` | Create DataLoaders | No |
| `validate_dataset.py` | Validate the data/model-input boundary | No |
| `vit_encoder.py` | Visual encoding + selective fine-tuning | No |
| `bert_encoder.py` | Text encoding + selective fine-tuning | No |
| `tabular_encoder.py` | Tabular MLP | No |
| `cross_attention.py` | Multimodal fusion | No |
| `task_heads.py` | Four prediction heads | No |
| `cinefusion_model.py` | Complete forward-pass architecture | No |
| `contrastive_loss.py` | Contrastive alignment loss | No |
| `model_losses.py` | Four supervised task losses | No |
| `phase2_losses.py` | Phase-2 loss composition | No |
| `train_phase_1.py` | Train one Phase-1 config on one train/validation split | No |
| `cross_validation.py` | TimeSeriesSplit over development data | No |
| `hyperp_tuning.py` | Optuna search and best-config selection | No |
| `train_final.py` | Final training on all development data | No |
| `train_phase_2.py` | Phase-2 alignment + joint optimization | No |
| `embedding_cache.py` | Cache frozen backbone/prefix representations | No |
| `calibration.py` | Temperature scaling + threshold finding | Never for test tuning |
| `uncertainty.py` | MC-Dropout uncertainty estimation | Optional at final evaluation |
| `metrics.py` | Centralized metric calculation | Yes, only for final evaluation |
| `evaluate.py` | Final inference/evaluation orchestration | Yes, only at the end |

---

# 18. Artifact Flow

## Development/model-selection artifacts

```text
phase1_best_config.json
    ↓
selected hyperparameter recipe

phase1_cv_results.json
    ↓
Optuna trial + fold history
```

## Model artifacts

```text
phase1_final.pt
    ↓
Final Phase-1 learned weights

phase2_final.pt
    ↓
Final Phase-2 learned weights
```

## Calibration artifacts

```text
phase2_calibration.json
    ↓
Temperature + learned genre thresholds
```

## Evaluation artifacts

```text
test_metrics.json
    ↓
Final test metrics

uncertainty outputs
    ↓
Optional MC-Dropout uncertainty results
```

Exact artifact filenames can be adjusted to the repository's path constants without changing their responsibilities.

---

# 19. Final End-to-End Pipeline

```text
┌──────────────────────────────────────────────────────────────┐
│                         DATA PREPARATION                      │
└──────────────────────────────────────────────────────────────┘

Raw Movie Data
      ↓
08_create_targets.py
      ↓
Task Targets + Masks
      ↓
09_split_dataset.py
      ↓
Development (<2019) ────────────────────── Test (>=2019)
      │                                           │
      ▼                                           │
preprocessing.py                                 │
      ↓                                           │
featurization.py                                 │
      ↓                                           │
custom_dataset.py                               │
      ↓                                           │
 data_loader.py                                  │
      ↓                                           │
validate_dataset.py                             │
      │                                           │
      ▼                                           │
┌──────────────────────────────────────────────────────────────┐
│                   SHARED MODEL ARCHITECTURE                   │
└──────────────────────────────────────────────────────────────┘

Poster → ViT → Visual Embedding
Overview → BERT → Text Embedding
Metadata → MLP → Tabular Embedding
                         │
                         ▼
                 Cross-Attention
                         │
                         ▼
                  Fused Representation
                         │
                         ▼
                     Task Heads
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
          Genre        Rating     Box Office
                           │
                           ▼
                    Content Rating

      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│                  PHASE 1: TASK-ONLY TRAINING                │
└──────────────────────────────────────────────────────────────┘

model_losses.py
      ↓
train_phase_1.py
      ↓
cross_validation.py
      ↓
TimeSeriesSplit
      ↓
hyperp_tuning.py
      ↓
Optuna
      ↓
Best Phase-1 Configuration
      ↓
train_final.py
      ↓
ALL Development Data
      ↓
Final Phase-1 Checkpoint

      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│          PHASE 2: ALIGNMENT + JOINT OPTIMIZATION            │
└──────────────────────────────────────────────────────────────┘

Best Phase-1 Checkpoint
      ↓
train_phase_2.py
      ├── existing model_losses.py
      └── contrastive_loss.py
      ↓
Selective backbone fine-tuning
      ↓
Final Phase-2 Checkpoint

      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│                  CALIBRATION / UNCERTAINTY                   │
└──────────────────────────────────────────────────────────────┘

Development Calibration Data
      ↓
calibration.py
      ├── Temperature Scaling
      └── Genre Threshold Finding
      ↓
LOCK calibration parameters

Final Model
      ↓
uncertainty.py
      ↓
Optional MC-Dropout uncertainty

      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│                     FINAL TEST EVALUATION                    │
└──────────────────────────────────────────────────────────────┘

Untouched 2019+ Test Data
      ↓
evaluate.py
      ↓
Apply locked calibration
      ↓
metrics.py
      ├── Genre F1 / Precision / Recall / ROC-AUC
      ├── Rating MAE / RMSE
      ├── Box-Office classification metrics / ROC-AUC
      └── Content-Rating classification metrics / ROC-AUC
      ↓
Final Phase-1 vs Phase-2 comparison
```

---

# 20. Design Principles

## Reuse the same model

Phase 2 does not duplicate the Phase-1 architecture.

The same:

```text
ViT
BERT
Tabular MLP
Cross-Attention
Task Heads
CineFusionModel
```

are reused.

## Reuse the same supervised loss

`model_losses.py` remains responsible for the four supervised tasks in both phases.

Phase 2 adds the contrastive objective through `contrastive_loss.py` and composes it through the Phase-2 loss layer where needed.

## Keep model selection away from the test set

Cross-validation, Optuna, Phase-2 development experiments, threshold finding, and temperature scaling operate on development data.

The test set is reserved for the final evaluation.

## Keep metric calculation centralized

`metrics.py` is the common implementation for development metrics and final test metrics.

It calculates performance; it does not tune the model.

## Keep calibration separate

`calibration.py` learns temperatures and thresholds.

`metrics.py` consumes those fixed values to evaluate predictions.

## Keep expensive backbone computation reusable

`embedding_cache.py` can cache frozen backbone sections without caching trainable projected embeddings.

---

# 21. One-Sentence Responsibility of Every Major Stage

```text
Data preparation
→ make leakage-safe multimodal samples.

Shared model
→ encode and fuse poster, text, and metadata.

Phase 1
→ learn the four supervised tasks.

Cross-validation + Optuna
→ choose the Phase-1 configuration using development data.

Final training
→ retrain the selected model using all development data.

Phase 2
→ improve multimodal alignment while continuing supervised learning.

Calibration
→ learn confidence scaling and genre decision thresholds from development data.

Uncertainty
→ estimate prediction variability with MC Dropout.

Metrics
→ measure model performance without changing model decisions.

Final evaluation
→ run the locked pipeline once on the untouched 2019+ test set.
```

---

# 22. Final Mental Model

CineFusion-X can be viewed as six layers:

```text
1. DATA
   ↓
   Turn movie records into clean, leakage-safe multimodal samples.

2. MODEL
   ↓
   Encode poster, text, and metadata and fuse them.

3. PHASE 1
   ↓
   Learn the four supervised movie tasks.

4. PHASE 2
   ↓
   Add explicit multimodal representation alignment while
   continuing supervised task learning.

5. CALIBRATION + UNCERTAINTY
   ↓
   Calibrate confidence, learn genre decision thresholds,
   and optionally estimate uncertainty.

6. FINAL EVALUATION
   ↓
   Evaluate the fully locked pipeline on the untouched
   future-period test set.
```

The complete conceptual architecture is:

```text
                    CINEFUSION-X
                         │
          ┌──────────────┼──────────────┐
          │              │              │
        Poster         Overview       Metadata
          │              │              │
         ViT            BERT           MLP
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                 Multimodal Attention
                         │
                         ▼
                 Fused Representation
                         │
          ┌──────────────┼──────────────┐
          │              │              │
        Genre          Rating       Box Office
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                  Content Rating

Phase 1:
    supervised multitask learning

Phase 2:
    supervised multitask learning
    + multimodal contrastive alignment

Post-training:
    temperature scaling
    + genre threshold finding
    + optional MC-Dropout uncertainty

Final:
    metrics on untouched 2019+ test data
```
