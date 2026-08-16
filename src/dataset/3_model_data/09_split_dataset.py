from pathlib import Path

import pandas as pd

from src.utils import print_section, save_dataframe


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("../data")

TARGET_DIR = (
    DATA_DIR
    / "processed"
    / "targets"
)

SPLIT_DIR = (
    DATA_DIR
    / "processed"
    / "splits"
)

MASTER_FILE = (
    TARGET_DIR
    / "multimodal_dataset_targets.csv"
)

BOX_OFFICE_FILE = (
    TARGET_DIR
    / "box_office_targets.csv"
)

CONTENT_RATING_FILE = (
    TARGET_DIR
    / "content_rating_targets.csv"
)


# ============================================================
# SPLIT SETTINGS
# ============================================================

CUTOFF_YEAR = 2019


# ============================================================
# MAIN
# ============================================================

def main():

    print_section("CHRONOLOGICAL DATASET SPLIT")

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 1. LOAD DATASETS
    # ========================================================

    master = pd.read_csv(
        MASTER_FILE,
        low_memory=False
    )

    box_office = pd.read_csv(
        BOX_OFFICE_FILE,
        low_memory=False
    )

    content_rating = pd.read_csv(
        CONTENT_RATING_FILE,
        low_memory=False
    )

    print(f"Master rows: {len(master):,}")
    print(f"Box-office rows: {len(box_office):,}")
    print(f"Content-rating rows: {len(content_rating):,}")

    # ========================================================
    # 2. VALIDATE REQUIRED COLUMNS
    # ========================================================

    required_master = {
        "imdb_id",
        "release_year"
    }

    required_box_office = {
        "imdb_id",
        "release_year",
        "box_office_target"
    }

    required_content_rating = {
        "imdb_id",
        "release_year",
        "content_rating_target"
    }

    if not required_master.issubset(master.columns):
        missing = required_master - set(master.columns)
        raise ValueError(
            f"Master dataset missing columns: {missing}"
        )

    if not required_box_office.issubset(box_office.columns):
        missing = required_box_office - set(box_office.columns)
        raise ValueError(
            f"Box-office dataset missing columns: {missing}"
        )

    if not required_content_rating.issubset(
        content_rating.columns
    ):
        missing = (
            required_content_rating
            - set(content_rating.columns)
        )

        raise ValueError(
            f"Content-rating dataset missing columns: {missing}"
        )

    # ========================================================
    # 3. CLEAN SPLIT KEYS
    # ========================================================

    for df in [
        master,
        box_office,
        content_rating
    ]:

        df["imdb_id"] = (
            df["imdb_id"]
            .astype("string")
            .str.strip()
        )

        df["release_year"] = pd.to_numeric(
            df["release_year"],
            errors="coerce"
        )

    # Master must have valid split information
    master = master[
        master["imdb_id"].notna()
        & master["release_year"].notna()
    ].copy()

    # ========================================================
    # 4. CREATE ONE MOVIE-LEVEL SPLIT
    # ========================================================

    development_ids = set(
        master.loc[
            master["release_year"] < CUTOFF_YEAR,
            "imdb_id"
        ]
    )

    test_ids = set(
        master.loc[
            master["release_year"] >= CUTOFF_YEAR,
            "imdb_id"
        ]
    )

    # ========================================================
    # 5. CHECK FOR DATA LEAKAGE
    # ========================================================

    overlap = development_ids.intersection(
        test_ids
    )

    if overlap:
        raise ValueError(
            f"Data leakage detected: "
            f"{len(overlap)} IMDb IDs appear in both splits."
        )

    print(
        f"\nDevelopment movies: "
        f"{len(development_ids):,}"
    )

    print(
        f"Test movies: "
        f"{len(test_ids):,}"
    )

    print(
        f"Development %: "
        f"{len(development_ids) / len(master) * 100:.2f}%"
    )

    print(
        f"Test %: "
        f"{len(test_ids) / len(master) * 100:.2f}%"
    )

    # ========================================================
    # 6. APPLY SAME SPLIT TO MASTER
    # ========================================================

    master_development = master[
        master["imdb_id"].isin(development_ids)
    ].copy()

    master_test = master[
        master["imdb_id"].isin(test_ids)
    ].copy()

    # ========================================================
    # 7. APPLY SAME SPLIT TO BOX OFFICE
    # ========================================================

    box_office_development = box_office[
        box_office["imdb_id"].isin(development_ids)
    ].copy()

    box_office_test = box_office[
        box_office["imdb_id"].isin(test_ids)
    ].copy()

    # ========================================================
    # 8. APPLY SAME SPLIT TO CONTENT RATING
    # ========================================================

    content_rating_development = content_rating[
        content_rating["imdb_id"].isin(development_ids)
    ].copy()

    content_rating_test = content_rating[
        content_rating["imdb_id"].isin(test_ids)
    ].copy()

    # ========================================================
    # 9. SAVE MASTER SPLITS
    # ========================================================

    save_dataframe(
        master_development,
        SPLIT_DIR / "master_development.csv"
    )

    save_dataframe(
        master_test,
        SPLIT_DIR / "master_test.csv"
    )

    # ========================================================
    # 10. SAVE BOX-OFFICE SPLITS
    # ========================================================

    save_dataframe(
        box_office_development,
        SPLIT_DIR / "box_office_development.csv"
    )

    save_dataframe(
        box_office_test,
        SPLIT_DIR / "box_office_test.csv"
    )

    # ========================================================
    # 11. SAVE CONTENT-RATING SPLITS
    # ========================================================

    save_dataframe(
        content_rating_development,
        SPLIT_DIR / "content_rating_development.csv"
    )

    save_dataframe(
        content_rating_test,
        SPLIT_DIR / "content_rating_test.csv"
    )

    # ========================================================
    # 12. TARGET DISTRIBUTION CHECKS
    # ========================================================

    print_section("POST-SPLIT TARGET CHECK")

    # --------------------------------------------------------
    # Box office
    # --------------------------------------------------------

    print("BOX OFFICE — DEVELOPMENT")

    print(
        box_office_development[
            "box_office_target"
        ]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nBOX OFFICE — TEST")

    print(
        box_office_test[
            "box_office_target"
        ]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    # --------------------------------------------------------
    # Content rating
    # --------------------------------------------------------

    print("\nCONTENT RATING — DEVELOPMENT")

    print(
        content_rating_development[
            "content_rating_target"
        ]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nCONTENT RATING — TEST")

    print(
        content_rating_test[
            "content_rating_target"
        ]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    print("\nRATING — DEVELOPMENT")

    print(
        master_development[
            "rating_target"
        ].describe()
    )

    print("\nRATING — TEST")

    print(
        master_test[
            "rating_target"
        ].describe()
    )

    # --------------------------------------------------------
    # Genre
    # --------------------------------------------------------

    development_genres = (
        master_development[
            "genre_target"
        ]
        .dropna()
        .astype("string")
        .str.split("|")
        .explode()
        .str.strip()
        .value_counts()
    )

    test_genres = (
        master_test[
            "genre_target"
        ]
        .dropna()
        .astype("string")
        .str.split("|")
        .explode()
        .str.strip()
        .value_counts()
    )

    print("\nGENRE — DEVELOPMENT")

    print(
        development_genres.head(10)
    )

    print("\nGENRE — TEST")

    print(
        test_genres.head(10)
    )

    # ========================================================
    # 13. FINAL VALIDATION
    # ========================================================

    print_section("SPLIT VALIDATION")

    print(
        "Master development rows:",
        len(master_development)
    )

    print(
        "Master test rows:",
        len(master_test)
    )

    print(
        "Box-office development rows:",
        len(box_office_development)
    )

    print(
        "Box-office test rows:",
        len(box_office_test)
    )

    print(
        "Content-rating development rows:",
        len(content_rating_development)
    )

    print(
        "Content-rating test rows:",
        len(content_rating_test)
    )

    # Verify chronological rule
    assert (
        master_development["release_year"] < CUTOFF_YEAR
    ).all()

    assert (
        master_test["release_year"] >= CUTOFF_YEAR
    ).all()

    # Verify no movie overlap
    assert set(
        master_development["imdb_id"]
    ).isdisjoint(
        set(master_test["imdb_id"])
    )

    print("\n✓ No movie overlap between development and test.")
    print("✓ Development contains movies released before 2019.")
    print("✓ Test contains movies released from 2019 onward.")
    print("✓ Same movie-level split applied to all task datasets.")
    print("✓ 5-fold CV will be performed later on development only.")

    print_section("SPLIT COMPLETE")


if __name__ == "__main__":
    main()