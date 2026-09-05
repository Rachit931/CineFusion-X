import os

import pandas as pd

import src.utils as utils
from config.paths import CHECKPOINT_DIR, POSTERS_DIR, TMDB_DIR

TMDB_METADATA = TMDB_DIR / "tmdb_metadata.csv"

VALIDATION_MARKER = CHECKPOINT_DIR / "tmdb_validation_complete.txt"


def validate_duplicates(metadata):

    print("\nChecking duplicate IDS....")

    imdb_duplicates = metadata["imdb_id"].duplicated().sum()

    tmdb_duplicates = metadata["tmdb_id"].duplicated().sum()

    print(f"Duplicate IMDb IDs : {imdb_duplicates:,}")
    print(f"Duplicate TMDB IDs {tmdb_duplicates:,}")


def validate_posters(metadata):

    print("\n Checking posters synchronizaton :::: ")

    metadata_ids = set(metadata["imdb_id"])

    poster_ids = set()

    for file in os.listdir(POSTERS_DIR):
        if file.endswith(".jpg"):
            poster_ids.add(file.replace(".jpg", ""))

    posters_without_metadata = poster_ids - metadata_ids

    metadata_without_posters = metadata_ids - poster_ids

    print(f"Poster files                    : {len(poster_ids):,}")

    print(f"Metadata rows                   : {len(metadata_ids):,}")

    print(f"Posters without metadata        : {len(posters_without_metadata):,}")

    print(f"Metadata without posters        : {len(metadata_without_posters):,}")

    if posters_without_metadata:
        print("\nExample posters without meatdata: ")

        print(sorted(list(posters_without_metadata))[:10])

    if metadata_without_posters:
        print("\nExample metadata without posters: ")

        print(sorted(list(metadata_without_posters))[:10])


def validate_missing_files(metadata):

    print("\nChecking missing poster files :::: ")

    missing = []

    for poster in metadata["poster_file"].dropna():
        if not os.path.exists(POSTERS_DIR / poster):
            missing.append(poster)

    print(f"Missing poster files : {len(missing):,}")

    if missing:
        print(missing[:10])


def validate_null_posters(metadata):

    print("\nChecking missing TMDB posters :::: ")

    null_paths = metadata["poster_path"].isna().sum()

    null_files = metadata["poster_file"].isna().sum()

    print(f"Null poster_path: {null_paths}")
    print(f"Null poster_file : {null_files}")


def main():

    utils.print_section("TMDB DATASET VALIDATION")

    metadata = pd.read_csv(TMDB_METADATA)

    validate_duplicates(metadata)

    validate_posters(metadata)

    validate_missing_files(metadata)

    validate_null_posters(metadata)

    with open(VALIDATION_MARKER, "w") as file:
        file.write("TMDB validation successfully completled \n")

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
