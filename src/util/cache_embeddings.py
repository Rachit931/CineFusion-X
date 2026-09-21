import argparse
import os

import numpy as np
import timm
import torch
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
from src.models.bert_encoder import BERTEncoder
from src.models.vit_encoder import ViTEncoder

# Environment

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

MASTER_PATHS = {
    "train": GENERAL_DIR / "master_training.csv",
    "test": GENERAL_DIR / "master_test.csv",
}

BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

MAX_TEXT_LENGTH = 256

BERT_MODEL = "bert-base-uncased"

# DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using PARSER in order to execute and save the embeddings based on
# which phase we are in, which is being done through a single file.

parser = argparse.ArgumentParser(description="Generate CineFusion-X ViT/BERT embedding cache")

parser.add_argument(
    "--mode",
    required=True,
    choices=["phase1_cache", "phase2_cache"],
    help="Cache type to generate.",
)

parser.add_argument(
    "--vit-blocks",
    type=int,
    default=None,
    help="Number of final ViT blocks that remain trainable in Phase 2.",
)

parser.add_argument(
    "--bert-layers",
    type=int,
    default=None,
    help="Number of final BERT layers that remain trainable in Phase 2.",
)

args = parser.parse_args()

CACHE_MODE = args.mode

# Phase 2 configuration

if CACHE_MODE == "phase2_cache":
    if args.vit_blocks is None:
        raise ValueError("--vit-blocks is required for phase2_cache.")

    if args.bert_layers is None:
        raise ValueError("--bert-layers is required for phase2_cache.")

    if args.vit_blocks <= 0:
        raise ValueError("--vit-blocks must be greater than 0.")

    if args.bert_layers <= 0:
        raise ValueError("--bert-layers must be greater than 0.")

    TRAINABLE_VIT_BLOCKS = args.vit_blocks
    TRAINABLE_BERT_LAYERS = args.bert_layers

else:
    TRAINABLE_VIT_BLOCKS = 0
    TRAINABLE_BERT_LAYERS = 0

# Cache root

if CACHE_MODE == "phase1_cache":
    CACHE_ROOT = PHASE1_CACHE_DIR

else:
    CACHE_ROOT = PHASE2_CACHE_DIR

# Creating the encoders


def create_encoders():
    """
    Create the ViT and BERT encoders using the exact
    Phase 1 / Phase 2 config
    """

    vit_encoder = ViTEncoder(output_dim=768, trainable_blocks=TRAINABLE_VIT_BLOCKS).to(DEVICE)

    bert_encoder = BERTEncoder(output_dim=768, trainable_layers=TRAINABLE_BERT_LAYERS).to(DEVICE)

    # Cache generation is inference only.
    vit_encoder.eval()
    bert_encoder.eval()

    return vit_encoder, bert_encoder


# Create preprocessing


def create_preprocessing(vit_encoder):
    """
    Create the same image preprocessing and BERT tokenizer
    used by MovieDataset.

    The preprocessing configuration is obtained directly from
    the already-created ViT encoder, so the ViT is not loaded twice.
    """

    vit_data_config = timm.data.resolve_model_data_config(vit_encoder.vit)

    vit_image_transform = timm.data.create_transform(
        **vit_data_config,
        is_training=False,
    )

    bert_tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL,
        token=hf_token,
    )

    return vit_image_transform, bert_tokenizer


# Generate cache for one split


def generate_cache_for_split(
    split_name,
    dataset,
    dataloader,
    vit_encoder,
    bert_encoder,
):
    """
    Generate a cache for one dataset split.

    Phase 1:
        ViT  : [N, 768]
        BERT : [N, 768]

    Phase 2:
        ViT  : [N, num_tokens, 768]
        BERT : [N, sequence_length, 768]

    Cached representations are converted to FP16 only when
    they are written to disk.
    """

    split_cache_dir = CACHE_ROOT / split_name

    split_cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vit_path = split_cache_dir / "vit_embeddings.npy"
    bert_path = split_cache_dir / "bert_embeddings.npy"
    imdb_ids_path = split_cache_dir / "imdb_ids.npy"

    attention_mask_path = split_cache_dir / "attention_mask.npy"

    # Saving the exact data point ordering used by this dataset.

    imdb_ids = dataset.data["imdb_id"].astype(str).to_numpy(dtype=str)

    np.save(
        imdb_ids_path,
        imdb_ids,
        allow_pickle=False,
    )

    vit_cache: np.memmap | None = None
    bert_cache: np.memmap | None = None
    attention_mask_cache: np.memmap | None = None

    total_samples = len(dataset)

    print(f"\nGenerating {CACHE_MODE} cache for {split_name} ({total_samples} movies)...")

    write_index = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(
            DEVICE,
            non_blocking=True,
        )

        input_ids = batch["input_ids"].to(
            DEVICE,
            non_blocking=True,
        )

        attention_mask = batch["attention_mask"].to(
            DEVICE,
            non_blocking=True,
        )

        with torch.inference_mode():
            if CACHE_MODE == "phase1_cache":
                vit_features = vit_encoder.get_phase1_cache(pixel_values)

                bert_features = bert_encoder.get_phase1_cache(input_ids, attention_mask)

            else:
                vit_features = vit_encoder.get_phase2_cache(pixel_values)

                bert_features = bert_encoder.get_phase2_cache(input_ids, attention_mask)

            # Converting the resulting representations to FP16 for
            # making it storage efficient

            vit_numpy = vit_features.detach().cpu().numpy().astype(np.float16, copy=False)

            bert_numpy = bert_features.detach().cpu().numpy().astype(np.float16, copy=False)

            # Creates output memory mapped arrays after the first batch

            if vit_cache is None:
                vit_cache = np.lib.format.open_memmap(
                    vit_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(
                        total_samples,
                        *vit_numpy.shape[1:],
                    ),
                )

                bert_cache = np.lib.format.open_memmap(
                    bert_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(
                        total_samples,
                        *bert_numpy.shape[1:],
                    ),
                )

                if CACHE_MODE == "phase2_cache":
                    attention_mask_cache = np.lib.format.open_memmap(
                        attention_mask_path,
                        mode="w+",
                        dtype=np.int64,
                        shape=(
                            total_samples,
                            attention_mask.shape[1],
                        ),
                    )

            # Determine where this batch belongs in the cache
            batch_start = write_index

            batch_end = batch_start + vit_numpy.shape[0]

            write_index = batch_end

            # Write ViT and BERT representations.

            assert vit_cache is not None
            assert bert_cache is not None

            vit_cache[batch_start:batch_end] = vit_numpy

            bert_cache[batch_start:batch_end] = bert_numpy

            # Save Phase 2 attention masks.

            if CACHE_MODE == "phase2_cache":
                assert attention_mask_cache is not None

                attention_mask_numpy = attention_mask.detach().cpu().numpy()

                attention_mask_cache[batch_start:batch_end] = attention_mask_numpy

            # Progress

            processed = batch_end
            percentage = (processed / total_samples) * 100

            print(
                f"\r{split_name} | {processed:,} / {total_samples:,} ({percentage:.2f}%)",
                end="",
                flush=True,
            )

    print(
        f"\r{split_name} | {processed} / {total_samples} points",
        end="",
        flush=True,
    )

    # Flush memory-mapped arrays to disk

    if vit_cache is not None:
        vit_cache.flush()

    if bert_cache is not None:
        bert_cache.flush()

    if attention_mask_cache is not None:
        attention_mask_cache.flush()

    print()

    print(f"Finished {CACHE_MODE} cache for {split_name}")
    print(f"  ViT:  {vit_path}")
    print(f"  BERT: {bert_path}")
    print(f"  IDs:  {imdb_ids_path}")

    if CACHE_MODE == "phase2_cache":
        print(f"  Mask: {attention_mask_path}")


def main():

    print("=" * 60)
    print("CineFusion-X Cache Generation")
    print("=" * 60)

    print(f"Cache mode: {CACHE_MODE}")
    print(f"Device: {DEVICE}")
    print(f"Cache root: {CACHE_ROOT}")

    if CACHE_MODE == "phase2_cache":
        print(f"Trainable ViT blocks: {TRAINABLE_VIT_BLOCKS}")
        print(f"Trainable BERT layers: {TRAINABLE_BERT_LAYERS}")

    # Create encoders once and reuse them for train + test.

    vit_encoder, bert_encoder = create_encoders()

    # Create preprocessing from the same ViT encoder.

    vit_image_transform, bert_tokenizer = create_preprocessing(vit_encoder)

    # Generate train and test caches separately.

    for split_name in ("train", "test"):
        master_path = MASTER_PATHS[split_name]

        dataset = MovieDataset(
            master_path=master_path,
            vit_image_transform=vit_image_transform,
            bert_tokenizer=bert_tokenizer,
            poster_dir=POSTERS_DIR,
            max_text_length=MAX_TEXT_LENGTH,
            cache_mode="normal",
        )

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
        )

        generate_cache_for_split(
            split_name=split_name,
            dataset=dataset,
            dataloader=dataloader,
            vit_encoder=vit_encoder,
            bert_encoder=bert_encoder,
        )

    print()
    print("=" * 60)
    print("Cache generation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
