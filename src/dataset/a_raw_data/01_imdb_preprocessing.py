import pandas as pd

from config.paths import IMDB_DIR

MIN_VOTES = 300


def preprocess_imdb():
    """
    Preprocess IMDB datasets and creates a clean movie dataset

    Input:
    1. title.basics.tsv
    2. title.ratings.tsv

    Output:
    imdb_movies_clean.csv
    """

    print("=" * 60)
    print("IMDb PREPROCESSING")
    print("=" * 60)

    # Loead IMDB datasets

    print("\nLoading IMDB datasets...")

    basics = pd.read_csv(
        IMDB_DIR / "title.basics.tsv",
        sep="\t",
        na_values="\\N",
        low_memory=False,
    )

    ratings = pd.read_csv(
        IMDB_DIR / "title.ratings.tsv",
        sep="\t",
        na_values="\\N",
        low_memory=False,
    )

    print(f"Original titles: {len(basics):,}")

    # Keep only movies (no other formats)

    movies = basics.loc[basics["titleType"] == "movie"].copy()

    print(f"Movies: {len(movies):,}")

    # Remove adult movies

    movies = movies.loc[movies["isAdult"] == 0]

    print(f"Non-adult movies: {len(movies):,}")

    # Keep only required columns

    movies = movies[
        [
            "tconst",
            "primaryTitle",
            "originalTitle",
            "startYear",
            "runtimeMinutes",
            "genres",
        ]
    ]

    # Remove missing values

    movies = movies.dropna(
        subset=[
            "startYear",
            "runtimeMinutes",
            "genres",
        ]
    )

    print(f"After removing missing values: {len(movies):,}")

    # Convert datatypes

    movies["startYear"] = movies["startYear"].astype(int)
    movies["runtimeMinutes"] = movies["runtimeMinutes"].astype(int)

    # Merge ratings

    imdb = movies.merge(
        ratings,
        on="tconst",
        how="inner",
    )

    print(f"After merging ratings: {len(imdb):,}")

    # Rename columns

    imdb = imdb.rename(
        columns={
            "tconst": "imdb_id",
            "primaryTitle": "title",
            "originalTitle": "original_title",
            "startYear": "release_year",
            "runtimeMinutes": "runtime_minutes",
            "averageRating": "rating",
            "numVotes": "num_votes",
        }
    )

    # Remove duplicate IMDB IDs

    imdb = imdb.drop_duplicates(subset="imdb_id")

    print(f"After removing duplicated: {len(imdb):,}")

    # Apply minimum vote threshold

    imdb = imdb.loc[imdb["num_votes"] >= MIN_VOTES]

    print(f"After vote threshold ({MIN_VOTES}): {len(imdb):,}")

    # Sort by popularity

    imdb = imdb.sort_values(
        by="num_votes",
        ascending=False,
    ).reset_index(drop=True)

    # Saved cleaned dataset

    output_path = IMDB_DIR / "imdb_movies_clean.csv"

    imdb.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print("\nSavied cleaned dataset: ")
    print(output_path)

    print("\nFinal dataset shape: ")
    print(imdb.shape)

    print("\nColumns: ")
    print(imdb.columns.tolist())

    print("\nIMDb preprocessing completed successfully.")

    return imdb


if __name__ == "__main__":
    preprocess_imdb()
