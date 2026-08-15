import pandas as pd  
from pathlib import Path

import src.utils as utils 

from config.paths import (
    MASTER_MULTIMODEL_DIR,
    TASK_HEADS_DIR,
    TARGETS_DIR,
)

MASTER_DATASET = MASTER_MULTIMODEL_DIR / "multimodel_dataset_prepared.csv"

BOX_OFFICE_DATASET = TASK_HEADS_DIR / "box_office_dataset.csv"

CONTENT_RATINGS_DATASET = TASK_HEADS_DIR / "content_ratings_dataset.csv"

MASTER_TARGET = TARGETS_DIR / "multimodel_dataset_targets.csv"

BOX_OFFICE_TARGET = TARGETS_DIR / "box_office_targets.csv"

CONTENT_RATINGS_TARGET = TARGETS_DIR / "content_ratings_targets.csv"

# BOX-OFFICE TARGET 

def create_box_office_target(df): 

    """
    Create a four-class box-office target using ROI: 

    ROI < 1       : FLOP
    1 <= ROI < 2  : AVERAGE
    2 <= ROI < 3  : HIT 
    3 <= ROI < 4  : BLOCKBUSTER
    """

    df =df.copy()

    # using ROI to cnostruct the target 

    df["roi"] = (
        (df["revenue"] - df["budget"])
        / df["budget"]
    ) * 100

    df["box_office_target"] = pd.cut(
        df["roi"], 
        bins=[
            float("-inf"),
            0,
            100,
            300,
            float("inf")
        ],
        labels=[
            "Flop",
            "Average",
            "Hit",
            "Blockbuster",
        ],
        right=False
    )

    return df

# CONTENT-RATING TARGET

def create_content_ratings_target(df): 
    """
    Keep only the four content-rating classes selected during EDA:
        
        G,
        PG,
        PG-13,
        R
    
    NR and NC-17 are excluded as a targets due to helplessness and small data points availability
    """

    df = df.copy()

    valid_rating = [
        "G",
        "PG",
        "PG-13",
        "R",
    ]

    df=df[
        df["content_rating"].isin(valid_rating)
    ].copy()

    df["content_rating_target"] = (
        df["content_rating"]
        .astype("string")
        .str.strip()
    )

    return df 

# RATINGS DATASET 

def create_ratings_target(df): 
    """
    Ratings is a continuous regression target
    """
    df=df.copy()

    df["rating_target"] = pd.to_numeric(
        df["rating"],
        errors = "coerce"
    )

    return df 

# GENRE TARGET 

def create_genre_target(df):
    """
    Genre is a multi-label target.

    Example: 
        Drama|Romance|Comedy 
    
    The raw multi-label string is kept here. 
    Actual multi-hot encoding happens after during preprocessing. 
    """

    df =df.copy()

    df["genre_target"] = (
        df["genres"]
        .astype("string")
        .str.strip()
    )

    return df


def main():

    utils.print_section("CREATE TARGETS")

    # Master dataset

    master = pd.read_csv(
        MASTER_DATASET,
        low_memory=False
    )

    print(f"Prepared master rows: {len(master):,}")

    # Rating target 

    master = create_ratings_target(master)

    print("Missing rating targets: ", master["rating_target"].isnull().sum())

    # Genre target 

    master = create_genre_target(master)

    print("Missing genre target: ", master["genre_target"].isna().sum())

    # Saving Master targets

    utils.save_dataframe(
        master,
        MASTER_TARGET
    )

    # Content-Rating target 

    content_rating = pd.read_csv(
        CONTENT_RATINGS_DATASET,
        low_memory=False
    )

    content_rating = create_content_ratings_target(
        content_rating
    )

    print(f"\nContent-rating rows: {len(content_rating):,}")

    print("\nContent-rating target distribution: ")

    print(
        content_rating[
            "content_rating_target"
        ]
        .value_counts()
        .sort_index()
    )

    print("\nContent-rating target percentages: ")
    print(
        (
            content_rating["content_rating_target"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    utils.save_dataframe(
        content_rating,
        CONTENT_RATINGS_TARGET
    )

    # Box-Office target 

    box_office = pd.read_csv(
        BOX_OFFICE_DATASET,
        low_memory=False
    )

    box_office = create_box_office_target(
        box_office
    )

    print(f"\nBox-office rows: {len(box_office):,}")

    print("\nBox-office target distribution: ")

    print(box_office[
        "box_office_target"
        ]
        .value_counts()
        .sort_index()
    )

    print("\nBox-office target percentages: ")

    print(
        (
            box_office["box_office_target"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    utils.save_dataframe(
        box_office,
        BOX_OFFICE_TARGET
    )

    # Summary 

    utils.print_section("TARGET CREATION COMPLETE")

    print("\n Saved files: ")

    print(MASTER_TARGET)
    print(CONTENT_RATINGS_TARGET)
    print(BOX_OFFICE_TARGET)


if __name__ == "__main__": 
    main()