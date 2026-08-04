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

    response = requests.get(
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

    reponse = requests.get(
        url,
        params=params,
        timeout=30,
    )

    if reponse.status_code != 200: 
        return None 

    return reponse.json()

def download_posters(
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

    response = requests.get(
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

