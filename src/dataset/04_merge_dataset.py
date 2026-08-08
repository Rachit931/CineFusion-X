import os 
import pandas as pd 

import src.utils as utils 

from config.paths import ( 
    IMDB_DIR,
    TMDB_DIR,
    POSTERS_DIR,
    MULTIMODEL_DATASET,
)

# Files 

IMDB_DATASET = IMDB_DIR / "imdb_movies_clean.csv"

TMDB_DATASET = TMDB_DIR / "tmdb_metadata.csv"

# Merging Datasets

def merge_datasets(
        imdb_df,
        tmdb_df,
):

    imdb_df = imdb_df[
        [

            "imdb_id",
            "release_year",
            "rating",
            "num_votes"
        ]
    ]   

    merged = imdb_df.merge(
        tmdb_df,
        on="imdb_id",
        how="inner",
    )

    return merged

# Removing points missing posters in poster_file

def removing_missing_posters(df): 

    df = df[df["poster_file"].notna()].copy()

    df = df[df["poster_file"] != ""].copy()

    return df.reset_index(drop = True)



# Removing points missing posters from POSTER_DIR even if poster_file exists

def removing_missing_poster_files(df): 

    missing = []

    for poster in df["poster_file"]: 

        poster_path = POSTERS_DIR / poster

        if not os.path.exists(
            poster_path
        ):

            missing.append(poster)

    if missing: 

        df = df[~df["poster_file"].isin(missing)].copy()

    return df.reset_index(drop = True)

def main():

    utils.print_section(
        "MERGING IMDb AND TMDB DATASET"
    )

    # Load IMDb dataset

    imdb_df = pd.read_csv(
        IMDB_DATASET
    )

    print(
        f"IMDb movies : {len(imdb_df):,}"
    )

    # Load TMDB metadata

    tmdb_df = pd.read_csv(
        TMDB_DATASET
    )

    print(
        f"TMDB movies : {len(tmdb_df):,}"
    )

    # Merge datasets

    merged = merge_datasets(
        imdb_df,
        tmdb_df,
    )

    print(
        f"Merged movies : {len(merged):,}"
    )

    # Remove movies without posters

    before = len(merged)

    merged = removing_missing_posters(
        merged
    )

    print(
        f"Movies without posters removed : "
        f"{before - len(merged):,}"
    )

    # Remove rows whose poster file
    # does not actually exist

    before = len(merged)

    merged = removing_missing_poster_files(
        merged
    )

    print(
        f"Missing poster files removed : "
        f"{before - len(merged):,}"
    )

    # Save final multimodal dataset

    utils.save_dataframe(
        merged,
        MULTIMODEL_DATASET,
    )

    # Final summary

    utils.print_section(
        "FINAL MULTIMODAL DATASET"
    )

    print(
        f"Movies : {len(merged):,}"
    )

    print(
        f"Columns : {len(merged.columns):,}"
    )

    print(
        "\nColumns:"
    )

    print(
        merged.columns.tolist()
    )

    print(
        f"\nSaved to:\n{MULTIMODEL_DATASET}"
    )


if __name__ == "__main__":

    main()