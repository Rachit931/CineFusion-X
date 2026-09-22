import os
from pathlib import Path

import timm
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.paths import (
    GENERAL_DIR,
    PHASE1_CACHE_DIR,
    PHASE2_CACHE_DIR,
    POSTERS_DIR,
)
from src.dataset.c_model_data.custom_dataset import MovieDataset


load_dotenv()

hf_token = os.getenv("HF_TOKEN")

MASTER_TRAIN = GENERAL_DIR / "master_training.csv"
MASTER_TEST = GENERAL_DIR / "master_test.csv"

VALID_CACHE_MODES = {
    "normal",
    "phase1_cache",
    "phase2_cache",
}

VALID_SPLITS = {
    "train",
    "test",
}


def _get_master_path(split):
    """
    Return the correct master dataset path for the requested split.
    """

    if split not in VALID_SPLITS:
        raise ValueError(
            f"split must be one of {VALID_SPLITS}, got '{split}'"
        )

    if split == "train":
        return MASTER_TRAIN

    return MASTER_TEST


def _get_cache_dir(cache_mode):
    """
    Return the cache directory corresponding to the requested cache mode.
    """

    if cache_mode == "normal":
        return None

    if cache_mode == "phase1_cache":
        return PHASE1_CACHE_DIR

    if cache_mode == "phase2_cache":
        return PHASE2_CACHE_DIR

    raise ValueError(
        f"cache_mode must be one of {VALID_CACHE_MODES}, "
        f"got '{cache_mode}'"
    )


def _create_normal_preprocessing():
    """
    Create the ViT image transform and BERT tokenizer.

    These are only required when raw images/text are being loaded.
    """

    # MIM-pretrained ViT-B/16
    vit_model_name = "vit_base_patch16_224.mae"

    vit_model = timm.create_model(
        vit_model_name,
        pretrained=True,
        num_classes=0,
    )

    # Resolve preprocessing configuration
    vit_data_config = timm.data.resolve_model_data_config(
        vit_model
    )

    vit_image_transform = timm.data.create_transform(
        **vit_data_config,
        is_training=False,
    )

    # BERT tokenizer
    bert_model_name = "bert-base-uncased"

    bert_tokenizer = AutoTokenizer.from_pretrained(
        bert_model_name,
        token=hf_token,
    )

    return vit_image_transform, bert_tokenizer


def create_dataset(
    cache_mode,
    split,
    max_text_length=256,
):
    """
    Create a MovieDataset for the requested pipeline stage.

    Parameters
    ----------
    cache_mode:
        "normal"
        "phase1_cache"
        "phase2_cache"

    split:
        "train"
        "test"

    max_text_length:
        Maximum BERT sequence length used in normal mode.
    """

    if cache_mode not in VALID_CACHE_MODES:
        raise ValueError(
            f"cache_mode must be one of {VALID_CACHE_MODES}, "
            f"got '{cache_mode}'"
        )

    master_path = _get_master_path(split)
    cache_dir = _get_cache_dir(cache_mode)

    # Defaults for cache modes.
    vit_image_transform = None
    bert_tokenizer = None

    # Raw mode requires actual preprocessing.
    if cache_mode == "normal":
        (
            vit_image_transform,
            bert_tokenizer,
        ) = _create_normal_preprocessing()

    dataset = MovieDataset(
        master_path=master_path,
        vit_image_transform=vit_image_transform,
        bert_tokenizer=bert_tokenizer,
        poster_dir=POSTERS_DIR,
        max_text_length=max_text_length,
        cache_mode=cache_mode,
        cache_dir=cache_dir,
        cache_split=split,
    )

    return dataset


def create_dataloader(
    cache_mode,
    split,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
):
    """
    Create a DataLoader for the requested pipeline stage.
    """

    dataset = create_dataset(
        cache_mode=cache_mode,
        split=split,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return loader