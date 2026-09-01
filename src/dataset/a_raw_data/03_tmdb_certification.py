import os 
import time 
import src.utils as utils

import pandas as pd 
import requests 
from dotenv import load_dotenv
from tqdm import tqdm 

from config.paths import (
    TMDB_DIR,
    CHECKPOINT_DIR
)

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise ValueError(
        "TMDB_API_KEY not found in .env file"
    )

TMDB_METADATA = TMDB_DIR / "tmdb_metadata.csv"

TMDB_METADATA_ENRICHED = TMDB_DIR / "tmdb_metadata_with_certi.csv"

CHECKPOINT_FILE = CHECKPOINT_DIR / "tmdb_certi_checkpoint.csv"

session = requests.Session()

def get_certifications(
        tmdb_id,
        api_key,
):

    url = (
    f"https://api.themoviedb.org/3/movie/"
    f"{tmdb_id}/release_dates"
    )    

    params = {
        "api_key" : api_key
    }

    for attempt in range(5): 

        try : 

            response = session.get(
                url,
                params=params,
                timeout=15,
            )

            if response.status_code == 200: 

                data = response.json()

                for country in data.get(
                    "results", []
                ):

                    if country.get(
                        "iso_3166_1"
                    ) != "US": 
                        continue

                    for release in country.get(
                        "release_dates",
                        []
                    ):

                        certification = release.get("certification")

                        if certification:
                            return certification.strip()

                    return None 

                return None 

            if response.status_code in {
                429, 500, 502, 503, 504, 
            }: 

                time.sleep(
                    2 ** attempt
                )

                continue 

            print ( 
                f"TMDB {tmdb_id}: "
                f"HTTP {response.status_code}"
            )

            return None 

        except requests.exceptions.RequestException as error: 

            if attempt == 4: 

                print(
                    f"TMDB {tmdb_id}: "
                    f"{error}"
                )

            time.sleep(
                2 ** attempt
            )

    return 

def collect_certification(
        tmdb_df,
        api_key,
): 

    if CHECKPOINT_FILE.exists():

        checkpoint = pd.read_csv(CHECKPOINT_FILE)

    else: 

        checkpoint = pd.DataFrame(
            columns = [
                "tmdb_id", 
                "content_rating",
            ]
        )

    completed = set(
        checkpoint[
            "tmdb_id"
        ].dropna().astype(int)
    )

    remaining_df = tmdb_df[
        ~tmdb_df["tmdb_id"]
        .astype("Int64")
        .isin(completed)
    ].reset_index(drop=True)

    print(
        f"Resuming from checkpoint: "
        f"Movies remaining: {len(remaining_df):,}"
    )

    records = checkpoint.to_dict("records")


    for tmdb_id in tqdm(
        remaining_df["tmdb_id"],
        desc="Collecting Certifications"
    ): 

        if pd.isna(tmdb_id):
            continue

        certification = get_certifications(
            int(tmdb_id),
            api_key
        )

        records.append({
            "tmdb_id": int(tmdb_id),
            "content_rating": certification,
        })

        # Checkpoint

        if len(records) % 500 == 0: 

            utils.save_checkpoint(
                records,
                CHECKPOINT_FILE,
            )

            print(
                f"Checkpoint saved: "
                f"{len(records):,} movies"
            )

    utils.save_checkpoint(
        records,
        CHECKPOINT_FILE,
    )

    certification_df = pd.DataFrame(records)

    enriched_df = tmdb_df.merge(
        certification_df[
            [
                "tmdb_id",
                "content_rating",
            ]
        ],
        on = "tmdb_id",
        how = "left",
    )

    return enriched_df


def main(): 

    utils.print_section("TMDB CERTIFICATIONS")

    tmdb_metadata = pd.read_csv(TMDB_METADATA)

    print(f"Loaded {len(tmdb_metadata):,} movies")

    if "tmdb_id" not in tmdb_metadata.columns: 

        raise ValueError(
            "tmdb_id column not found in tmdb_metadata.csv"
        )

    tmdb_metadata = collect_certification(
        tmdb_metadata,
        TMDB_API_KEY,
    )

    utils.save_dataframe(
        tmdb_metadata,
        TMDB_METADATA_ENRICHED
    )

    total_movies = len(
        tmdb_metadata
    )

    certified_movies = (
        tmdb_metadata[
            "content_rating"
        ]
        .notna()
        .sum()
    )   

    print(f"\nTotal movies : {total_movies}:,")

    print(f"Certifications found: {certified_movies}:,")

    print(f"Coverage: {certified_movies / total_movies * 100: .2f}%")


    print("\nCertification Distribution:")

    print(tmdb_metadata[
        "content_rating"
        ].value_counts(
            dropna=False
        )
    )

    print(f"\nSaved to: {TMDB_METADATA_ENRICHED}")


if __name__ == "__main__":

    main()
