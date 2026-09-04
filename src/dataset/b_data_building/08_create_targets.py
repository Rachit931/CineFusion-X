import pandas as pd

import src.utils as utils
from config.paths import (
    MASTER_MULTIMODEL_DIR,
    TARGETS_DIR,
    TASK_HEADS_DIR,
)

MASTER_DATASET = MASTER_MULTIMODEL_DIR / "multimodel_dataset_prepared.csv"

BOX_OFFICE_DATASET = TASK_HEADS_DIR / "box_office_dataset.csv"

CONTENT_RATINGS_DATASET = TASK_HEADS_DIR / "content_ratings_dataset.csv"

MASTER_TARGET = TARGETS_DIR / "multimodel_dataset_targets.csv"

BOX_OFFICE_TARGET = TARGETS_DIR / "box_office_targets.csv"

CONTENT_RATINGS_TARGET = TARGETS_DIR / "content_ratings_targets.csv"


# TARGET CLASS DEFINITIONS

GENRE_CLASSES = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "TV Movie",
    "Thriller",
    "War",
    "Western",
]


BOX_OFFICE_CLASSES = {
    "Flop": 0,
    "Average": 1,
    "Hit": 2,
    "Blockbuster": 3,
}


CONTENT_RATING_CLASSES = {
    "G": 0,
    "PG": 1,
    "PG-13": 2,
    "R": 3,
}


# BOX-OFFICE TARGET


def create_box_office_target(df):
    """
    Create a four-class box-office target using ROI:

    ROI < 0%          : FLOP
    0% <= ROI < 100%  : AVERAGE
    100% <= ROI < 300%: HIT
    ROI >= 300%       : BLOCKBUSTER
    """

    df = df.copy()

    # using ROI to construct the target

    df["roi"] = ((df["revenue"] - df["budget"]) / df["budget"]) * 100

    df["box_office_target"] = pd.cut(
        df["roi"],
        bins=[
            float("-inf"),
            0,
            100,
            300,
            float("inf"),
        ],
        labels=[
            "Flop",
            "Average",
            "Hit",
            "Blockbuster",
        ],
        right=False,
    )

    # Convert class names into class IDs

    df["box_office_target"] = df["box_office_target"].map(BOX_OFFICE_CLASSES).astype("Int64")

    return df


# CONTENT-RATING TARGET


def create_content_ratings_target(df):
    """
    Keep only the four content-rating classes selected during EDA:

        G,
        PG,
        PG-13,
        R

    NR and NC-17 are excluded as targets due to
    limited data availability.
    """

    df = df.copy()

    valid_rating = [
        "G",
        "PG",
        "PG-13",
        "R",
    ]

    df = df[df["content_rating"].isin(valid_rating)].copy()

    # Convert class names into class IDs

    df["content_rating_target"] = (
        df["content_rating"]
        .astype("string")
        .str.strip()
        .map(CONTENT_RATING_CLASSES)
        .astype("Int64")
    )

    return df


# RATINGS DATASET


def create_ratings_target(df):
    """
    Ratings is a continuous regression target.
    """

    df = df.copy()

    df["rating_target"] = pd.to_numeric(
        df["rating"],
        errors="coerce",
    )

    return df


# GENRE TARGET


def create_genre_target(df):
    """
    Genre is a multi-label target.

    Example:
        Drama|Romance|Comedy

    Each genre gets its own binary target column.

    Example:

        genre_action_target
        genre_adventure_target
        genre_animation_target
        ...
        genre_western_target

    A movie can have multiple genres, so these
    targets are multi-hot encoded.
    """

    df = df.copy()

    # Clean and split the genres for each movie

    movie_genres = (
        df["genres"]
        .fillna("")
        .astype(str)
        .apply(lambda value: {genre.strip() for genre in value.split("|") if genre.strip()})
    )

    # Create one binary target column per genre

    for genre in GENRE_CLASSES:
        column_name = "genre_" + genre.lower().replace(" ", "_") + "_target"

        df[column_name] = movie_genres.apply(
            lambda genres, genre=genre: int(genre in genres)
        ).astype("int8")

    return df


def main():

    utils.print_section("CREATE TARGETS")

    # Master dataset

    master = pd.read_csv(
        MASTER_DATASET,
        low_memory=False,
    )

    master = master.copy()

    print(f"Prepared master rows: {len(master):,}")

    # Rating target

    master = create_ratings_target(master)

    print(
        "Missing rating targets: ",
        master["rating_target"].isnull().sum(),
    )

    # Genre target

    master = create_genre_target(master)

    genre_target_columns = [
        column
        for column in master.columns
        if column.startswith("genre_") and column.endswith("_target")
    ]

    print(
        "\nGenre target columns: ",
        len(genre_target_columns),
    )

    print(genre_target_columns)

    # Content-Rating target

    content_rating = pd.read_csv(
        CONTENT_RATINGS_DATASET,
        low_memory=False,
    )

    content_rating = create_content_ratings_target(content_rating)

    print(f"\nContent-rating rows: {len(content_rating):,}")

    print("\nContent-rating target distribution: ")

    print(content_rating["content_rating_target"].value_counts().sort_index())

    print("\nContent-rating target percentages: ")

    print(
        (
            content_rating["content_rating_target"].value_counts(normalize=True).sort_index() * 100
        ).round(2)
    )

    # Add content rating target to MASTER

    master = master.merge(
        content_rating[
            [
                "imdb_id",
                "content_rating_target",
            ]
        ],
        on="imdb_id",
        how="left",
    )

    # Box-Office target

    box_office = pd.read_csv(
        BOX_OFFICE_DATASET,
        low_memory=False,
    )

    box_office = create_box_office_target(box_office)

    print(f"\nBox-office rows: {len(box_office):,}")

    print("\nBox-office target distribution: ")

    print(box_office["box_office_target"].value_counts().sort_index())

    print("\nBox-office target percentages: ")

    print(
        (box_office["box_office_target"].value_counts(normalize=True).sort_index() * 100).round(2)
    )

    # Add box office target

    master = master.merge(
        box_office[
            [
                "imdb_id",
                "box_office_target",
            ]
        ],
        on="imdb_id",
        how="left",
    )

    # Save target datasets

    utils.save_dataframe(
        master,
        MASTER_TARGET,
    )

    utils.save_dataframe(
        content_rating,
        CONTENT_RATINGS_TARGET,
    )

    utils.save_dataframe(
        box_office,
        BOX_OFFICE_TARGET,
    )

    # Summary

    utils.print_section("TARGET CREATION COMPLETE")

    print("\nSaved files:")

    print(MASTER_TARGET)
    print(CONTENT_RATINGS_TARGET)
    print(BOX_OFFICE_TARGET)


if __name__ == "__main__":
    main()
