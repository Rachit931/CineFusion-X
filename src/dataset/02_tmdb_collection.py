import os 
import time 

import pandas as pd 
import requests 

from dotenv import load_dotenv 

from config.paths import (
    IMDB_DIR,
    TMDB_DIR,
    POSTERS_DIR,
)

# Load Environment Variables 

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY: 
    raise ValueError(
        "TMDB_API_KEY not found in .env"
    )

# TMDB URLs

TMDB_BASE_URL = "https://api.themoviedb.org/3"

POSTER_BASE_URL = "https://image.tmdb.org/t/p/original"

# Input / Output Files 

IMDB_DATASET = (
    IMDB_DIR / 
    "imdb_movies_clean.csv"
)

TMDB_METADATA = ( 
    TMDB_DIR / 
    "tmdb_metadata.csv"
)

# Load IMDb dataset 

print("=" * 60)
print("TMDB DATA COLLECTION")
print("=" * 60)

print("\nLoading IMDb dataset...")

imdb = pd.read_csv(IMDB_DATASET)

print(f"Movies loaded: {len(imdb):,}")

# Resume Previous Run 

if TMDB_METADATA.exists(): 

    existing = pd.read_csv(TMDB_METADATA)

    completed_ids = set(existing["imdb_id"])

    imdb = imdb[
        ~imdb["imdb_id"].isin(
            completed_ids
        )
    ].reset_index(drop=True)

    metadata = existing.to_dict("records")

    print(f"Already collected: {len(existing):,}")

    print(f"Remaining movies: {len(imdb):,}")

else : 

    metadata = []

# TMDB Request function 

def get_tmdb_movie(imdb_id): 
    """
    Fetch movie metadata from TMDB 
    using IMDb id 
    """

    url = f"{TMDB_BASE_URL}/find/{imdb_id}"

    params = {
        "api_key": TMDB_API_KEY,
        "external_source": "imdb_id",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    if response.status_code != 200: 
        return None 

    results = response.json()

    movies = results.get(
        "movie_results",
        []
    )

    if len(movies) == 0: 
        return None 

    return movies[0]

