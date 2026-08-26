from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from config.paths import (
    GENERAL_DIR,
    POSTERS_DIR,
)

from src.dataset.c_model_data.custom_dataset import MovieDataset
from src.dataset.c_model_data.data_loader import (
    vit_image_transform,
    bert_tokenizer
)

# ============================================================
# PATHS
# ============================================================

MASTER_TRAINING_FILE = (
    GENERAL_DIR / "master_training.csv"
)

BATCH_SIZE = 16
MAX_TEXT_LENGTH = 512


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("CINEFUSION-X DATASET VALIDATION")
    print("=" * 80)


    # ========================================================
    # 1. LOAD FINAL TRAINING DATASET
    # ========================================================

    print("\n[1] LOADING FINAL DATASET")
    print("-" * 80)

    df = pd.read_csv(
        MASTER_TRAINING_FILE,
        low_memory=False,
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    assert len(df) > 0, (
        "Master training dataset is empty."
    )

    print(
        "✓ Master training dataset loaded."
    )


    # ========================================================
    # 2. FIND GENRE TARGET COLUMNS
    # ========================================================

    print("\n[2] GENRE TARGET STRUCTURE")
    print("-" * 80)

    genre_target_columns = [
        column
        for column in df.columns
        if column.startswith("genre_")
        and column.endswith("_target")
    ]

    print(
        f"Genre target columns: "
        f"{len(genre_target_columns)}"
    )

    assert len(genre_target_columns) == 19, (
        f"Expected 19 genre target columns, "
        f"found {len(genre_target_columns)}."
    )

    print(
        "✓ Exactly 19 genre target columns found."
    )


    # ========================================================
    # 3. REQUIRED TARGETS
    # ========================================================

    print("\n[3] REQUIRED TARGETS")
    print("-" * 80)

    required_targets = [
        "rating_target",
        "box_office_target",
        "content_rating_target",
        *genre_target_columns,
    ]

    for column in required_targets:

        assert column in df.columns, (
            f"Missing target column: {column}"
        )

    print(
        "✓ All four task targets are present."
    )


    # ========================================================
    # 4. OLD GENRE TARGET MUST NOT EXIST
    # ========================================================

    print("\n[4] OLD GENRE TARGET CHECK")
    print("-" * 80)

    assert "genre_target" not in df.columns, (
        "Old single-column genre_target still exists."
    )

    print(
        "✓ Old genre_target column is absent."
    )


    # ========================================================
    # 5. GENRE MULTI-HOT VALIDATION
    # ========================================================

    print("\n[5] GENRE MULTI-HOT VALIDATION")
    print("-" * 80)

    genre_values = (
        df[genre_target_columns]
        .to_numpy()
    )

    non_binary_values = genre_values[
        ~np.isin(
            genre_values,
            [0, 1],
        )
    ]

    assert len(non_binary_values) == 0, (
        "Genre target contains values other than 0/1."
    )

    print(
        "✓ All genre target values are 0 or 1."
    )


    # ========================================================
    # 6. GENRE LABEL COUNT
    # ========================================================

    genre_label_count = (
        df[genre_target_columns]
        .sum(axis=1)
    )

    zero_genre = (
        genre_label_count == 0
    ).sum()

    valid_genre = (
        genre_label_count > 0
    ).sum()

    print(
        f"Movies with >=1 genre : {valid_genre:,}"
    )

    print(
        f"Movies with 0 genres  : {zero_genre:,}"
    )


    # ========================================================
    # 7. RATING TARGET
    # ========================================================

    print("\n[6] RATING TARGET")
    print("-" * 80)

    rating_values = pd.to_numeric(
        df["rating_target"],
        errors="coerce",
    )

    valid_rating = rating_values.dropna()

    assert (
        valid_rating >= 0
    ).all(), (
        "Negative rating target found."
    )

    assert (
        valid_rating <= 10
    ).all(), (
        "Rating target above 10 found."
    )

    print(
        f"Valid ratings: {len(valid_rating):,}"
    )

    print(
        f"Missing ratings: "
        f"{rating_values.isna().sum():,}"
    )

    print(
        "✓ Rating targets are within 0–10."
    )


    # ========================================================
    # 8. BOX OFFICE TARGET
    # ========================================================

    print("\n[7] BOX-OFFICE TARGET")
    print("-" * 80)

    box_values = pd.to_numeric(
        df["box_office_target"],
        errors="coerce",
    )

    valid_box = box_values.dropna()

    assert set(
        valid_box.unique()
    ).issubset(
        {0, 1, 2, 3}
    ), (
        "Invalid box-office class found."
    )

    print(
        "Observed classes:",
        sorted(valid_box.unique())
    )

    print(
        f"Missing box-office targets: "
        f"{box_values.isna().sum():,}"
    )

    print(
        "✓ Box-office classes are valid."
    )


    # ========================================================
    # 9. CONTENT RATING TARGET
    # ========================================================

    print("\n[8] CONTENT-RATING TARGET")
    print("-" * 80)

    content_values = pd.to_numeric(
        df["content_rating_target"],
        errors="coerce",
    )

    valid_content = (
        content_values.dropna()
    )

    assert set(
        valid_content.unique()
    ).issubset(
        {0, 1, 2, 3}
    ), (
        "Invalid content-rating class found."
    )

    print(
        "Observed classes:",
        sorted(valid_content.unique())
    )

    print(
        f"Missing content-rating targets: "
        f"{content_values.isna().sum():,}"
    )

    print(
        "✓ Content-rating classes are valid."
    )


    # ========================================================
    # 10. TARGET / FEATURE LEAKAGE
    # ========================================================

    print("\n[9] TARGET LEAKAGE CHECK")
    print("-" * 80)

    excluded_columns = [
        "imdb_id",
        "overview",
        "rating_target",
        "box_office_target",
        "content_rating_target",
        *genre_target_columns,
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    leaked_targets = [
        column
        for column in required_targets
        if column in feature_columns
    ]

    assert not leaked_targets, (
        f"Target leakage detected: "
        f"{leaked_targets}"
    )

    print(
        "✓ No target columns are present in MLP features."
    )

    print(
        f"MLP feature count: "
        f"{len(feature_columns)}"
    )


    # ========================================================
    # 11. FEATURE DATA TYPES
    # ========================================================

    print("\n[10] FEATURE DATA TYPES")
    print("-" * 80)

    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    assert not non_numeric_features, (
        f"Non-numeric MLP features: "
        f"{non_numeric_features}"
    )

    print(
        "✓ All MLP features are numeric."
    )


    # ========================================================
    # 12. FEATURE NaN / INF
    # ========================================================

    print("\n[11] FEATURE NaN / INF CHECK")
    print("-" * 80)

    feature_array = df[
        feature_columns
    ].to_numpy(
        dtype=np.float64
    )

    nan_count = np.isnan(
        feature_array
    ).sum()

    inf_count = np.isinf(
        feature_array
    ).sum()

    print(
        f"NaN count : {nan_count:,}"
    )

    print(
        f"Inf count : {inf_count:,}"
    )

    assert nan_count == 0, (
        "NaN values found in MLP features."
    )

    assert inf_count == 0, (
        "Infinite values found in MLP features."
    )

    print(
        "✓ MLP features contain no NaN/Inf."
    )


    # ========================================================
    # 13. IMDb ID VALIDATION
    # ========================================================

    print("\n[12] IMDb ID VALIDATION")
    print("-" * 80)

    assert "imdb_id" in df.columns

    assert df[
        "imdb_id"
    ].notna().all(), (
        "Missing IMDb IDs found."
    )

    assert (
        df["imdb_id"].nunique()
        == len(df)
    ), (
        "Duplicate IMDb IDs found."
    )

    print(
        "✓ IMDb IDs are present and unique."
    )


    # ========================================================
    # 14. POSTER VALIDATION
    # ========================================================

    print("\n[13] POSTER VALIDATION")
    print("-" * 80)

    missing_posters = []

    for imdb_id in df["imdb_id"]:

        poster_path = (
            POSTERS_DIR
            / f"{imdb_id}.jpg"
        )

        if not poster_path.exists():

            missing_posters.append(
                imdb_id
            )

    print(
        f"Missing posters: "
        f"{len(missing_posters):,}"
    )

    assert not missing_posters, (
        f"{len(missing_posters)} posters "
        f"are missing."
    )

    print(
        "✓ Every movie has a poster."
    )


    # ========================================================
    # 15. CREATE MOVIEDATASET
    # ========================================================

    print("\n[14] MOVIEDATASET VALIDATION")
    print("-" * 80)

    dataset = MovieDataset(
        master_path=MASTER_TRAINING_FILE,
        vit_image_transform=vit_image_transform,
        bert_tokenizer=bert_tokenizer,
        poster_dir=POSTERS_DIR,
        max_text_length=MAX_TEXT_LENGTH,
    )

    assert len(dataset) == len(df), (
        "MovieDataset length does not match CSV."
    )

    print(
        f"✓ MovieDataset loaded: "
        f"{len(dataset):,} samples"
    )


    # ========================================================
    # 16. CHECK SINGLE SAMPLE
    # ========================================================

    print("\n[15] SINGLE SAMPLE CHECK")
    print("-" * 80)

    sample = dataset[0]

    expected_keys = {
        "imdb_id",

        "pixel_values",
        "input_ids",
        "attention_mask",
        "features",

        "genre_target",
        "rating_target",
        "box_office_target",
        "content_rating_target",

        "genre_mask",
        "rating_mask",
        "box_office_mask",
        "content_rating_mask",
    }

    assert set(
        sample.keys()
    ) == expected_keys, (
        "MovieDataset returned unexpected keys."
    )

    print(
        "✓ Sample contains exactly expected fields."
    )


    # ========================================================
    # 17. INPUT TENSOR CHECK
    # ========================================================

    print("\n[16] INPUT TENSOR CHECK")
    print("-" * 80)

    assert torch.is_tensor(
        sample["pixel_values"]
    )

    assert torch.is_tensor(
        sample["input_ids"]
    )

    assert torch.is_tensor(
        sample["attention_mask"]
    )

    assert torch.is_tensor(
        sample["features"]
    )

    assert (
        sample["pixel_values"].dtype
        == torch.float32
    )

    assert (
        sample["input_ids"].dtype
        == torch.long
    )

    assert (
        sample["attention_mask"].dtype
        == torch.long
    )

    assert (
        sample["features"].dtype
        == torch.float32
    )

    print(
        "✓ All model inputs are tensors "
        "with correct dtypes."
    )


    # ========================================================
    # 18. TARGET TENSOR CHECK
    # ========================================================

    print("\n[17] TARGET TENSOR CHECK")
    print("-" * 80)

    for target_name in [
        "genre_target",
        "rating_target",
        "box_office_target",
        "content_rating_target",
    ]:

        assert torch.is_tensor(
            sample[target_name]
        ), (
            f"{target_name} is not a tensor."
        )

    assert (
        sample["genre_target"].dtype
        == torch.float32
    )

    assert (
        sample["rating_target"].dtype
        == torch.float32
    )

    assert (
        sample["box_office_target"].dtype
        == torch.long
    )

    assert (
        sample["content_rating_target"].dtype
        == torch.long
    )

    print(
        "✓ All targets are tensors "
        "with correct dtypes."
    )


    # ========================================================
    # 19. TARGET SHAPE CHECK
    # ========================================================

    print("\n[18] TARGET SHAPE CHECK")
    print("-" * 80)

    assert (
        sample["genre_target"].shape
        == (19,)
    )

    assert (
        sample["rating_target"].ndim
        == 0
    )

    assert (
        sample["box_office_target"].ndim
        == 0
    )

    assert (
        sample["content_rating_target"].ndim
        == 0
    )

    print(
        "✓ Single-sample target shapes are correct."
    )


    # ========================================================
    # 20. MASK CHECK
    # ========================================================

    print("\n[19] MASK CHECK")
    print("-" * 80)

    for mask_name in [
        "genre_mask",
        "rating_mask",
        "box_office_mask",
        "content_rating_mask",
    ]:

        assert torch.is_tensor(
            sample[mask_name]
        ), (
            f"{mask_name} is not a tensor."
        )

        assert (
            sample[mask_name].dtype
            == torch.bool
        ), (
            f"{mask_name} is not torch.bool."
        )

    print(
        "✓ All masks are torch.bool tensors."
    )


    # ========================================================
    # 21. GENRE MASK CHECK
    # ========================================================

    expected_genre_mask = (
        sample["genre_target"].sum()
        > 0
    )

    assert (
        bool(sample["genre_mask"])
        == bool(expected_genre_mask)
    ), (
        "Genre mask does not match "
        "genre target."
    )

    print(
        "✓ Genre mask correctly matches "
        "genre target."
    )


    # ========================================================
    # 22. DATALOADER
    # ========================================================

    print("\n[20] DATALOADER CHECK")
    print("-" * 80)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    batch = next(
        iter(loader)
    )

    print(
        f"Batch size: "
        f"{batch['features'].shape[0]}"
    )

    assert (
        batch["features"].shape[0]
        == BATCH_SIZE
    )

    print(
        "✓ DataLoader successfully created a batch."
    )


    # ========================================================
    # 23. BATCH TARGET SHAPES
    # ========================================================

    print("\n[21] BATCH TARGET SHAPES")
    print("-" * 80)

    assert (
        batch["genre_target"].shape
        == (BATCH_SIZE, 19)
    )

    assert (
        batch["rating_target"].shape
        == (BATCH_SIZE,)
    )

    assert (
        batch["box_office_target"].shape
        == (BATCH_SIZE,)
    )

    assert (
        batch["content_rating_target"].shape
        == (BATCH_SIZE,)
    )

    print(
        "✓ Batch target shapes are correct."
    )


    # ========================================================
    # 24. BATCH MASK SHAPES / TYPES
    # ========================================================

    print("\n[22] BATCH MASK CHECK")
    print("-" * 80)

    for mask_name in [
        "genre_mask",
        "rating_mask",
        "box_office_mask",
        "content_rating_mask",
    ]:

        assert (
            batch[mask_name].dtype
            == torch.bool
        )

        assert (
            batch[mask_name].shape
            == (BATCH_SIZE,)
        )

    print(
        "✓ Batch masks are boolean tensors "
        "with correct shapes."
    )


    # ========================================================
    # 25. GENRE BATCH VALIDATION
    # ========================================================

    print("\n[23] GENRE BATCH VALIDATION")
    print("-" * 80)

    assert torch.all(
        (batch["genre_target"] == 0)
        | (batch["genre_target"] == 1)
    )

    expected_genre_masks = (
        batch["genre_target"]
        .sum(dim=1)
        > 0
    )

    assert torch.equal(
        batch["genre_mask"],
        expected_genre_masks,
    )

    print(
        "✓ Genre targets are multi-hot."
    )

    print(
        "✓ Genre masks match genre targets."
    )


    # ========================================================
    # 26. RATING MASK VALIDATION
    # ========================================================

    print("\n[24] RATING MASK VALIDATION")
    print("-" * 80)

    valid_ratings = (
        batch["rating_target"][
            batch["rating_mask"]
        ]
    )

    assert torch.all(
        (valid_ratings >= 0)
        & (valid_ratings <= 10)
    )

    print(
        "✓ Every unmasked rating target "
        "is valid."
    )


    # ========================================================
    # 27. BOX-OFFICE MASK VALIDATION
    # ========================================================

    print("\n[25] BOX-OFFICE MASK VALIDATION")
    print("-" * 80)

    valid_box = (
        batch["box_office_target"][
            batch["box_office_mask"]
        ]
    )

    masked_box = (
        batch["box_office_target"][
            ~batch["box_office_mask"]
        ]
    )

    assert torch.all(
        (valid_box >= 0)
        & (valid_box <= 3)
    )

    assert torch.all(
        masked_box == -1
    )

    print(
        "✓ Unmasked box-office targets "
        "are valid."
    )

    print(
        "✓ Masked box-office targets "
        "contain only placeholder -1."
    )


    # ========================================================
    # 28. CONTENT-RATING MASK VALIDATION
    # ========================================================

    print("\n[26] CONTENT-RATING MASK VALIDATION")
    print("-" * 80)

    valid_content = (
        batch["content_rating_target"][
            batch["content_rating_mask"]
        ]
    )

    masked_content = (
        batch["content_rating_target"][
            ~batch["content_rating_mask"]
        ]
    )

    assert torch.all(
        (valid_content >= 0)
        & (valid_content <= 3)
    )

    assert torch.all(
        masked_content == -1
    )

    print(
        "✓ Unmasked content-rating targets "
        "are valid."
    )

    print(
        "✓ Masked content-rating targets "
        "contain only placeholder -1."
    )


    # ========================================================
    # 29. CRITICAL MASKING TEST
    # ========================================================

    print("\n[27] CRITICAL MASKING TEST")
    print("-" * 80)

    # This verifies that selecting by mask NEVER selects
    # a missing target.

    selected_box = (
        batch["box_office_target"][
            batch["box_office_mask"]
        ]
    )

    selected_content = (
        batch["content_rating_target"][
            batch["content_rating_mask"]
        ]
    )

    selected_rating = (
        batch["rating_target"][
            batch["rating_mask"]
        ]
    )

    selected_genre = (
        batch["genre_target"][
            batch["genre_mask"]
        ]
    )

    assert torch.all(
        (selected_box >= 0)
        & (selected_box <= 3)
    )

    assert torch.all(
        (selected_content >= 0)
        & (selected_content <= 3)
    )

    assert torch.all(
        (selected_rating >= 0)
        & (selected_rating <= 10)
    )

    assert torch.all(
        selected_genre.sum(dim=1) > 0
    )

    print(
        "✓ Mask=True selects only valid targets."
    )


    # ========================================================
    # 30. NO NaN / INF IN BATCH
    # ========================================================

    print("\n[28] BATCH NaN / INF CHECK")
    print("-" * 80)

    for name in [
        "pixel_values",
        "features",
        "genre_target",
        "rating_target",
    ]:

        assert torch.isfinite(
            batch[name]
        ).all(), (
            f"NaN/Inf found in {name}."
        )

    print(
        "✓ No NaN/Inf in numerical batch values."
    )


    # ========================================================
    # 31. FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("🟢 ALL DATASET VALIDATION PASSED")
    print("=" * 80)

    print(
        """
CSV
✓ Final training dataset loads
✓ 19 genre targets
✓ Genre targets are multi-hot
✓ Rating target is valid
✓ Box-office classes are valid
✓ Content-rating classes are valid
✓ No target leakage
✓ Features are numeric
✓ Features contain no NaN/Inf
✓ IMDb IDs are valid
✓ Posters exist

MOVIEDATASET
✓ __getitem__ works
✓ Inputs are tensors
✓ Targets are tensors
✓ Target dtypes are correct
✓ Target shapes are correct
✓ Masks are boolean
✓ Genre masking is correct

DATALOADER
✓ Batch creation works
✓ Batch shapes are correct
✓ Batch dtypes are correct
✓ Batch masks are correct
✓ Mask=True selects only valid targets
✓ No NaN/Inf in numerical batch values

READY FOR MODELING.
"""
    )


if __name__ == "__main__":
    main()