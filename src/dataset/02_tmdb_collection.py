import os 
import time 

from pathlib import Path 

import pandas as pd 
import requests
from dotenv import load_dotenv
from tqdm import tqdm 

from config.paths import ( 
    IMDB_DIR,
    TMDB_DIR,
    POSTERS_DIR,
)

session = requests.Session()
# Load Environment Variables 

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY: 
    raise ValueError(
        "TMDB_API_KEY not found in .env"
    )

# Files 

IMDB_DATASET = (
    IMDB_DIR / 
    "imdb_movies_clean.csv"
)

TMDB_METADATA = (
    TMDB_DIR / 
    "tmdb_metadata.csv"
)

CHECKPOINT_FILE = (
    TMDB_DIR / 
    "tmdb_metadata_checkpoint.csv"
)

POSTERS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TMDB_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# TMDB Functions

def get_tmdb_id(
        imdb_id,
        api_key,
): 
    
    url = (
        f"https://api.themoviedb.org/3/find/{imdb_id}"
    )

    params = {
        "api_key": api_key,
        "external_source": "imdb_id",
    }

    response = session.get(
        url,
        params=params,
        timeout=30,
    )

    if response.status_code != 200: 
        return None 

    data = response.json()

    if len(data["movie_results"]) == 0: 
        return None

    return data["movie_results"][0]["id"]


def get_movie_details(
        tmdb_id, 
        api_key,
): 

    url = (
        f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    )

    params = {
        "api_key": api_key
    }

    reponse = session.get(
        url,
        params=params,
        timeout=30,
    )

    if reponse.status_code != 200: 
        return None 

    return reponse.json()

def download_poster(
        imdb_id,
        poster_path,
): 

    if not poster_path: 
        return None 

    poster_url = ( 
        "https://image.tmdb.org/t/p/original"
        + poster_path
    )

    poster_file = (f"{imdb_id}.jpg")

    save_path = (
        POSTERS_DIR / 
        poster_file
    )

    # Skip download if already exists 

    if save_path.exists(): 
        return poster_file 

    response = session.get(
        poster_url,
        timeout=60,
    )

    if response.status_code != 200:
        return None 

    with open(
        save_path,
        "wb",
    ) as file: 

        file.write(
            response.content
        )

    return poster_file 

# Collect TMDB Metadata 

def collect_tmdb_metadata( 
        imdb_df,
        api_key,
): 

    all_movies = []

    # Resume from checkpoint 

    if CHECKPOINT_FILE.exists(): 

        checkpoint = pd.read_csv(
            CHECKPOINT_FILE
        )

        all_movies = checkpoint.to_dict(
            "records"
        )

        completed = set(
            checkpoint["imdb_id"]
        )

        imdb_df = imdb_df[
            ~imdb_df["imdb_id"].isin(
                completed
            )
        ].reset_index(drop=True)

        print(
            f"Resuming from checkpoint "
            f"({len(completed):,} movies already processed)"
        )

    # Main Loop 

    for i, imdb_id in enumerate(
        tqdm(
            imdb_df["imdb_id"],
            desc = "Collecting TMDB Metadata"
        )
    ): 

        tmdb_id = get_tmdb_id(
            imdb_id,
            api_key
        )

        if tmdb_id is None:
            print(f"No TMDB ID: {imdb_id}")
            continue 

        print(f"Found TMDB ID: {tmdb_id}")

        details = get_movie_details(
            tmdb_id,
            api_key
        )

        if details is None: 
            print(f"Failed details: {imdb_id}")
            continue 

        print(f"Got details: {imdb_id}")

        # Clean complex fields 

        genres = "|".join(

            genre["name"]

            for genre in details.get(
                "genres",
                [],
            )
        )

        production_companies = "|".join(

            company["name"]

            for company in details.get(
                "production_companies",
                [],
            )
        )

        production_countries = "|".join(

            country["name"]

            for country in details.get(
                "production_countries",
                [],
            )
        )

        spoken_languages = "|".join(

            language["english_name"]

            for language in details.get(
                "spoken_languages",
                [],
            )
        )

        # Downloads Poster 

        poster_file = download_poster(

            imdb_id,
            details.get(
                "poster_path"
            )
        )

        # Store Metadata 

        movies_data = {
            "imdb_id": imdb_id,

            "tmdb_id": details["id"],

            "title": details.get("title"),

            "original_title": details.get("original_title"),

            "overview": details.get("overview"),

            "tagline": details.get("tagline"),

            "poster_path": details.get("poster_path"),

            "poster_file": poster_file,

            "budget": details.get("budget"),

            "revenue": details.get("revenue"),

            "runtime": details.get("runtime"),

            "popularity": details.get("popularity"),

            "release_date": details.get('release_date'),

            "original_language": details.get("original_language"),

            "vote_average": details.get("vote_average"), 

            "vote_count": details.get("vote_count"),

            "genres": genres,

            "production_companies": production_companies,

            "production_countries": production_countries,

            "spoken_languages": spoken_languages,
        }

        all_movies.append(
            movies_data
        )

        # Save checkpoint every 500 movies 

        if ( 
            len(all_movies) % 500 == 0
        ): 

            checkpoint = pd.DataFrame(all_movies)

            checkpoint.to_csv(
                CHECKPOINT_FILE,
                index = False,
            )

            print(
                f"Checkpoint saved",
                f"({len(all_movies):,} movies)"
            )


    # Final Metadata 

    tmdb_metadata = pd.DataFrame(
        all_movies
    )

    tmdb_metadata.to_csv(
        TMDB_METADATA,
        index=False,
    )

    return tmdb_metadata 

# MAIN 

def main(): 

    print("=" * 60) 
    print("TMDB METADATA COLLECTION")
    print("=" * 60)

    imdb_movies = pd.read_csv(IMDB_DATASET)

    tmdb_metadata = collect_tmdb_metadata(
        imdb_movies,
        TMDB_API_KEY,
    )

    print(
        f"\nCollected",
        f"{len(tmdb_metadata):,} movies."
    
    )

    print(f"\nSaved to:\n{TMDB_METADATA}")

if __name__ == "__main__": 

    main()