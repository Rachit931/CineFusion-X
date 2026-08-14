import pandas as pd
from pathlib import Path

import src.utils as utils

from config.paths import ( 
    BASE_DIR,
    MASTER_MULTIMODEL_DIR,
    TASK_HEADS_DIR,
)

INPUT_DATASET = MASTER_MULTIMODEL_DIR / "multimodel_dataset_enriched.csv"

MASTER_DATASET = MASTER_MULTIMODEL_DIR / "multimodel_dataset_prepared.csv"

BOX_OFFICE_DATASET = TASK_HEADS_DIR / "box_office_dataset.csv"

CONTENT_RATINGS_DATASET = TASK_HEADS_DIR / "content_ratings_dataset.csv"


def main():

    utils.print_section("DATASET PREPARATION")

    df = pd.read_csv(
            INPUT_DATASET,
            low_memory=False,
    )

    print(f"Input rows         : {len(df)}")
    print(f"Input columns      : {len(df.columns)}")

    # Basic identity validation 

    # EDA confirmed:
    # - no missing IMDb IDs
    # - no duplicate IMDb IDs
    # - no malformed IMDb IDs
    
    # Therefore, no rows are removed here.

    missing_ids = df["imdb_id"].isna().sum()

    duplicate_ids = df["imdb_id"].duplicated().sum()

    print(f"Missing IMDb IDs   : {missing_ids}")
    print(f"Duplicate IMDb IDs : {duplicate_ids}")

    # Convert financial columns to numeric

    df["budget"] = pd.to_numeric(
        df["budget"],
        errors = "coerce"
    )

    df["revenue"] = pd.to_numeric(
        df["revenue"],
        errors = "coerce"
    )

    # Save prepared master dataset

    # No general row removal is performed because the EDA
    # did not identify another confirmed invalid-row rule.
    # Missing feature values remain for later transformation 
    # in later preproccessing.

    utils.save_dataframe(
        df,
        MASTER_DATASET
    )

    # Create box-office dataset

    # EDA decision:
    
    # budget < 100  -> invalid for box-office task
    # revenue < 100 -> invalid for box-office task
    # Both must be >= 100.

    box_office_df = df[
        (df["budget"] >= 100)
        & (df["revenue"] >= 100)
    ].copy()

    # Save box office dataset

    utils.save_dataframe(
        box_office_df,
        BOX_OFFICE_DATASET
    )

    # 6. Content-rating candidate dataset
    
    # Final EDA rule:
    # Keeping only:
    # G, PG, PG-13, R
    
    # NR    : not an actual rating class
    # NC-17 : too few examples (204)

    valid_ratings = [
        "G",
        "PG",
        "PG-13",
        "R",
    ]

    content_ratings_df = df[
        df["content_rating"].isin(valid_ratings)
    ].copy()

    # Save content_ratings_dataset

    utils.save_dataframe(
        box_office_df,
        CONTENT_RATINGS_DATASET,
    )

    # REPORT 

    utils.print_section("PREPARATION RESULTS")

    print(f"Prepared master dataset : {len(df)} rows")
    

    print(f"Box-office dataset      : {len(box_office_df)} rows")
    

    print(f"Content-rating dataset  : {len(content_ratings_df):,}")

    print(
        f"Excluded from box-office: "
        f"{len(df) - len(box_office_df)} rows"
    )

    print("\nBox-office validation:")

    print(
        "Budget < 100 :",
        (box_office_df["budget"] < 100).sum()
    )

    print(
        "Revenue < 100:",
        (box_office_df["revenue"] < 100).sum()
    )

    print(
        "Missing budget :",
        box_office_df["budget"].isna().sum()
    )

    print(
        "Missing revenue:",
        box_office_df["revenue"].isna().sum()
    )

    print("\nContent-rating classes: ")

    print(
        content_ratings_df["content_rating"]
        .value_counts()
        .to_string()
    )

    print("\nSaved:")

    print(MASTER_DATASET)
    print(BOX_OFFICE_DATASET)


if __name__ == "__main__":
    main()