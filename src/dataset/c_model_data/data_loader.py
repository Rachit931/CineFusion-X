import os

import timm
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.paths import GENERAL_DIR, PHASE1_CACHE_DIR, PHASE2_CACHE_DIR, POSTERS_DIR
from src.dataset.c_model_data.custom_dataset import (
    MovieDataset,
)

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

MASTER_TRAIN = GENERAL_DIR / "master_training.csv"
MASTER_TEST = GENERAL_DIR / "master_test.csv"

# Cache configuration

CACHE_MODE = "normal"

VALID_CACHE_MODES = {
    "normal",
    "phase1_cache",
    "phase2_cache",
}

if CACHE_MODE not in VALID_CACHE_MODES:
    raise ValueError(f"CACHE_MODE must be one of {VALID_CACHE_MODES}, instead got '{CACHE_MODE}")

# Selecting the phase-specific cache directory
if CACHE_MODE == "phase1_cache":
    CACHE_DIR = PHASE1_CACHE_DIR

elif CACHE_MODE == "phase2_cache":
    CACHE_DIR = PHASE2_CACHE_DIR

else:
    CACHE_DIR = None


# Only required below code for normal mode.
# In cache modes, the ViT/BERT reprepresentations are already cached
# through the cache_embeddings file.

if CACHE_MODE == "normal":
    # MIM-pretrained ViT-B/16
    VIT_MODEL = "vit_base_patch16_224.mae"

    vit_model = timm.create_model(
        VIT_MODEL,
        pretrained=True,
        num_classes=0,
    )

    # Expected prprocessing for image as per the model chosen
    vit_data_config = timm.data.resolve_model_data_config(vit_model)

    vit_image_transform = timm.data.create_transform(
        **vit_data_config,
        is_training=False,
    )

    # Creating BERT's tokenizer by loading it from the pretrained model
    BERT_MODEL = "bert-base-uncased"

    # And loads all the processing required for the text to be given input into our BERT
    # by applying BERT's tokenizers as processing
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, token=hf_token)


# Create train dataset

train_dataset = MovieDataset(
    MASTER_TRAIN,
    vit_image_transform,
    bert_tokenizer,
    POSTERS_DIR,
    max_text_length=256,
    cache_mode=CACHE_MODE,
    cache_dir=CACHE_DIR,
    cache_split="train",
)

# Create test dataset

test_dataset = MovieDataset(
    MASTER_TEST,
    vit_image_transform,
    bert_tokenizer,
    POSTERS_DIR,
    max_text_length=256,
    cache_mode=CACHE_MODE,
    cache_dir=CACHE_DIR,
    cache_split="test",
)

# Create training loader for pytorch

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
