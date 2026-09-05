import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variable from .env

load_dotenv()

# Root directory containing the dataset

DATA_ROOT = Path(os.getenv("CINEFUSION_DATA_ROOT", "./data")).expanduser().resolve()

PROJECT_ROOT = Path(__file__).expanduser().resolve()

# RAW DATA

RAW_DIR = DATA_ROOT / "raw"

IMDB_DIR = RAW_DIR / "imdb"
TMDB_DIR = RAW_DIR / "tmdb"
POSTERS_DIR = RAW_DIR / "posters"

EXTERNAL_DIR = RAW_DIR / "external"

# PROCESSED DATA

PROCESSED_DIR = DATA_ROOT / "processed"

BASE_DIR = PROCESSED_DIR / "interim"
MASTER_MULTIMODEL_DIR = PROCESSED_DIR / "master"
TASK_HEADS_DIR = PROCESSED_DIR / "tasks"
TARGETS_DIR = PROCESSED_DIR / "targets"

SPLITS_DIR = PROCESSED_DIR / "splits"

S_BOX_OFFICE_SPLIT_DIR = SPLITS_DIR / "box_office"
S_CONTENT_RATING_DIR = SPLITS_DIR / "content_rating"
S_GENERAL_DIR = SPLITS_DIR / "general"

CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints"

# FEATURIZED DATA

FEATURE_DIR = DATA_ROOT / "features"

BOX_OFFICE_DIR = FEATURE_DIR / "box_office"
CONTENT_RATING_DIR = FEATURE_DIR / "content_rating"
GENERAL_DIR = FEATURE_DIR / "general"

# LOGS

LOG_DIR = DATA_ROOT / "logs"

# ARTIFACTS

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

PREPROCESSOR_DIR = ARTIFACTS_DIR / "preprocessor"

# Automatically create folders

DIRECTORIES = [
    IMDB_DIR,
    TMDB_DIR,
    POSTERS_DIR,
    BASE_DIR,
    MASTER_MULTIMODEL_DIR,
    TASK_HEADS_DIR,
    TARGETS_DIR,
    SPLITS_DIR,
    S_BOX_OFFICE_SPLIT_DIR,
    S_CONTENT_RATING_DIR,
    S_GENERAL_DIR,
    FEATURE_DIR,
    BOX_OFFICE_DIR,
    CONTENT_RATING_DIR,
    GENERAL_DIR,
    CHECKPOINT_DIR,
    FEATURE_DIR,
    LOG_DIR,
    PREPROCESSOR_DIR,
    ARTIFACTS_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)
