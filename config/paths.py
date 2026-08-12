from pathlib import Path 
import os 

from dotenv import load_dotenv 

# Load environment variable from .env 

load_dotenv()

# Root directory containing the dataset
 
DATA_ROOT = Path(
    os.getenv("CINEFUSION_DATA_ROOT", "./data")
).expanduser().resolve()

# RAW DATA 

RAW_DIR = DATA_ROOT / "raw"

IMDB_DIR = RAW_DIR / "imdb"
TMDB_DIR = RAW_DIR / "tmdb"
POSTERS_DIR = RAW_DIR / "posters"

# PROCESSED DATA 

PROCESSED_DIR = DATA_ROOT / "processed"

IMAGE_DIR = PROCESSED_DIR / "image"
TEXT_DIR = PROCESSED_DIR / "text"
TABULAR_DIR = PROCESSED_DIR / "tabular"

MULTIMODEL_DIR = PROCESSED_DIR / "multimodel_data"

CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints"

# LOGS 

LOG_DIR = DATA_ROOT / "logs"

# Automatically create folders 

DIRECTORIES = [
    IMDB_DIR,
    TMDB_DIR,
    POSTERS_DIR,
    IMAGE_DIR,
    TEXT_DIR,
    TABULAR_DIR,
    MULTIMODEL_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
]

for directory in DIRECTORIES: 
    directory.mkdir(parents=True, exist_ok=True)