from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    """
    CineFusion-X moive dataset.

    It supports three states:

    1. Loads poster images and tokenized BERT inputs.
        (heavily unoptimized)

    2. Phase1_cache
        Load cached final ViT and BERT backbone representations.
        Only the CLS token of each.

        ViT:   [768]
        BERT:  [768]

    3. Phase2_cache
        Load cached representation immediately before the
        trainable ViT blocks / BERT layers.

        ViT: [num_tokens, 768]

        BERT: [sequence_length, 768]
    """

    def __init__(
        self,
        master_path,
        vit_image_transform=None,
        bert_tokenizer=None,
        poster_dir=None,
        max_text_length=256,
        cache_mode="normal",
        cache_dir=None,
        cache_split="train",
    ):

        super().__init__()

        # Cache model validation

        valid_cache_modes = {
            "normal",
            "phase1_cache",
            "phase2_cache",
        }

        if cache_mode not in valid_cache_modes:
            raise ValueError(
                f"cache_mode must be one of the {valid_cache_modes}, instead got '{cache_mode}"
            )

        self.cache_mode = cache_mode
        self.cache_split = cache_split

        # Load the featurized dataset
        self.data = pd.read_csv(
            master_path,
            low_memory=False,
        )

        # Store Preprocessors
        self.image_processor = vit_image_transform
        self.tokenizer = bert_tokenizer
        self.poster_dir = Path(poster_dir) if poster_dir is not None else None

        # Validation for normal mode

        if self.cache_mode == "normal":
            if self.image_processor is None:
                raise ValueError("vit_image_transform is required in normal mode.")

            if self.tokenizer is None:
                raise ValueError("bert_tokenizer is required in normal mode.")

            if self.poster_dir is None:
                raise ValueError("poster_dir is required in normal mode.")

        # finding all genre target columns
        self.genre_target_columns = [
            column
            for column in self.data.columns
            if column.startswith("genre_") and column.endswith("_target")
        ]

        if len(self.genre_target_columns) != 19:
            raise ValueError(
                f"Expected 19 genre target columns, found {len(self.genre_target_columns)}"
            )

        # Features not meant to be used for MLP
        excluded_columns = [
            "imdb_id",
            "overview",
            "rating_target",
            "box_office_target",
            "content_rating_target",
            *self.genre_target_columns,
        ]

        # Select only requried preprocessed features
        self.feature_columns = [
            column for column in self.data.columns if column not in excluded_columns
        ]

        # conversion into tensors
        self.features = torch.tensor(
            self.data[self.feature_columns].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )

        # Targets

        self.genre_targets = torch.tensor(
            self.data[self.genre_target_columns].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )

        self.rating_targets = torch.tensor(
            self.data["rating_target"].fillna(0).to_numpy(dtype="float32"),
            dtype=torch.float32,
        )

        self.box_office_targets = torch.tensor(
            self.data["box_office_target"].fillna(-1).to_numpy(dtype="int64"),
            dtype=torch.long,
        )

        self.content_rating_targets = torch.tensor(
            self.data["content_rating_target"].fillna(-1).to_numpy(dtype="int64"),
            dtype=torch.long,
        )

        # Target masks

        self.genre_masks = self.genre_targets.sum(dim=1) > 0

        self.rating_masks = torch.tensor(
            self.data["rating_target"].notna().to_numpy(),
            dtype=torch.bool,
        )

        self.box_office_masks = torch.tensor(
            self.data["box_office_target"].notna().to_numpy(),
            dtype=torch.bool,
        )

        self.content_rating_masks = torch.tensor(
            self.data["content_rating_target"].notna().to_numpy(),
            dtype=torch.bool,
        )

        # FOR "normal" MODE ONLY:

        # Basic cleaning of overviews
        if self.cache_mode == "normal":
            overview = self.data["overview"].fillna("").astype(str).tolist()

            # Tokenizing all the overview text at once
            tokenized = self.tokenizer(
                overview,
                padding="max_length",
                truncation=True,
                max_length=max_text_length,
                return_tensors="pt",
            )

            # Store BERT inputs
            self.input_ids = tokenized["input_ids"]

            self.attention_mask = tokenized["attention_mask"]

        else:
            self.input_ids = None
            self.attention_mask = None

        # Load cache files for cache modes

        if self.cache_mode != "normal":
            if cache_dir is None:
                raise ValueError("cache_dir is required when cache_mode is not 'normal'")

            self.cache_dir: Path | None = Path(cache_dir)

            self._load_cache_files()

        else:
            self.cache_dir = None

            self.vit_cache: np.ndarray | None = None
            self.bert_cache: np.ndarray | None = None
            self.cache_attention_mask: np.ndarray | None = None

    # Cache Loading

    def _load_cache_files(self):
        """
        Load cache arrays(embeddings) using Numpy memory mapping.

        This is neccessary for phase 2 because the complete cache size
        will be easily 30+ GB. Not optimal to load the whole cache into the
        ram at once.
        """

        phase_directory = self.cache_dir / self.cache_split

        if not phase_directory.exists():
            raise FileNotFoundError(f"Cache directory not found: {phase_directory}")

        imdb_ids_path = phase_directory / "imdb_ids.npy"

        vit_path = phase_directory / "vit_embeddings.npy"

        bert_path = phase_directory / "bert_embeddings.npy"

        if not imdb_ids_path.exists():
            raise FileNotFoundError(f"Missing cache file: {imdb_ids_path}")

        if not vit_path.exists():
            raise FileNotFoundError(f"Missing cache file: {vit_path}")

        if not bert_path.exists():
            raise FileNotFoundError(f"Missing cache file: {bert_path}")

        # Memory-mapped loading.

        # The full cache is not copied completely into the RAM.

        self.cache_imdb_ids = np.load(
            imdb_ids_path,
            mmap_mode="r",
            allow_pickle=True,
        )

        self.vit_cache = np.load(
            vit_path,
            mmap_mode="r",
        )

        self.bert_cache = np.load(bert_path, mmap_mode="r")

        self.cache_attention_mask = None

        # Phase 2 additionally needs the BERT attention mask.
        if self.cache_mode == "phase2_cache":
            attention_mask_path = phase_directory / "attention_mask.npy"

            if not attention_mask_path.exists():
                raise FileNotFoundError(f"Missing Phase 2 cache file: {attention_mask_path}")

            self.cache_attention_mask = np.load(
                attention_mask_path,
                mmap_mode="r",
            )

        self._validate_cache()

    # Cache validation

    def _validate_cache(self):
        """
        Verifying thatt the caches belongs to exactly this dataset
        and that the dimensions matches with the dataset size.
        """

        assert self.vit_cache is not None
        assert self.bert_cache is not None

        if self.cache_mode == "phase2_cache":
            assert self.cache_attention_mask is not None

        # For comparison

        dataset_ids = self.data["imdb_id"].astype(str).to_numpy()

        cache_ids = np.asarray(self.cache_imdb_ids).astype(str)

        # Number of points must match

        if len(cache_ids) != len(self.data):
            raise ValueError(
                "Cache size does not match dataset size. "
                f"Dataset: {len(self.data)}, "
                f"Cache: {len(cache_ids)}."
            )

        # Movie ordering must match.

        if not np.array_equal(
            cache_ids,
            dataset_ids,
        ):
            raise ValueError(
                "Cache movie ordering does not match the dataset. "
                "The cache must be generated from the same dataset "
                "in the same row order."
            )

        # ViT / BERT first dimension

        if self.vit_cache.shape[0] != len(self.data):
            raise ValueError("ViT cache first dimension does not match the dataset size.")

        if self.bert_cache.shape[0] != len(self.data):
            raise ValueError("BERT cache first dimension does not match the dataset size.")

        # Expected embedding dimension

        if self.vit_cache.shape[-1] != 768:
            raise ValueError(
                f"Expected ViT cache hidden dimension of 768, got {self.vit_cache.shape[-1]}."
            )

        if self.bert_cache.shape[-1] != 768:
            raise ValueError(
                f"Expected BERT cache hidden dimension of 768, got {self.bert_cache.shape[-1]}."
            )

        # Phase 1 shape validation

        if self.cache_mode == "phase1_cache":
            if self.vit_cache.ndim != 2:
                raise ValueError("Phase 1 ViT cache must have shape [N, 768].")

            if self.bert_cache.ndim != 2:
                raise ValueError("Phase 1 BERT cache must have shape [N, 768].")

        # Phase 2 shape validation

        if self.cache_mode == "phase2_cache":
            assert self.vit_cache is not None
            assert self.bert_cache is not None
            assert self.cache_attention_mask is not None

            if self.vit_cache.ndim != 3:
                raise ValueError("Phase 2 ViT cache must have shape [N, num_tokens, 768].")

            if self.bert_cache.ndim != 3:
                raise ValueError("Phase 2 BERT cache must have shape [N, sequence_length, 768].")

            if self.cache_attention_mask.shape[0] != len(self.data):
                raise ValueError(
                    "Phase 2 attention-mask cache size does not match the dataset size."
                )

            if self.cache_attention_mask.shape[1] != self.bert_cache.shape[1]:
                raise ValueError(
                    "Phase 2 attention-mask sequence length "
                    "does not match BERT cache sequence length."
                )

    # Dataset Size

    def __len__(self):

        return len(self.data)

    # Fetching data points as per index

    def __getitem__(self, index):

        row = self.data.iloc[index]

        # imdb identifier
        imdb_id = row["imdb_id"]

        # Tabular input
        features = self.features[index]

        # Targets
        genre_target = self.genre_targets[index]

        rating_target = self.rating_targets[index]

        box_office_target = self.box_office_targets[index]

        content_rating_target = self.content_rating_targets[index]

        # Target masks

        genre_mask = self.genre_masks[index]

        rating_mask = self.rating_masks[index]

        box_office_mask = self.box_office_masks[index]

        content_rating_mask = self.content_rating_masks[index]

        # NORMAL MODE WITHOUT CACHING

        if self.cache_mode == "normal":
            assert self.poster_dir is not None

            # Matching poster
            # matching poster as per the imdb_id
            poster_path = self.poster_dir / f"{imdb_id}.jpg"

            if not poster_path.exists():
                raise FileNotFoundError(f"Poster not found for the {imdb_id}{poster_path}")

            # Load Poster
            image = Image.open(poster_path).convert("RGB")

            # Process images for ViT
            pixel_values = self.image_processor(image)

            # BERT inputs
            input_ids = self.input_ids[index]

            attention_mask = self.attention_mask[index]

            # Return one movie
            return {
                "imdb_id": imdb_id,
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "features": features,
                "genre_target": genre_target,
                "rating_target": rating_target,
                "box_office_target": box_office_target,
                "content_rating_target": content_rating_target,
                "genre_mask": genre_mask,
                "rating_mask": rating_mask,
                "box_office_mask": box_office_mask,
                "content_rating_mask": content_rating_mask,
            }

        # PHASE 1 CACHE MODE

        if self.cache_mode == "phase1_cache":
            # Read only this movie's cached embeddings.
            # The full cache remains memory-mapped on disk.

            assert self.vit_cache is not None
            assert self.bert_cache is not None

            vit_embedding = torch.from_numpy(np.array(self.vit_cache[index], copy=True))

            bert_embedding = torch.from_numpy(np.array(self.bert_cache[index], copy=True))

            return {
                "imdb_id": imdb_id,
                "cached_visual_embedding": vit_embedding,
                "cached_text_embedding": bert_embedding,
                "features": features,
                "genre_target": genre_target,
                "rating_target": rating_target,
                "box_office_target": box_office_target,
                "content_rating_target": content_rating_target,
                "genre_mask": genre_mask,
                "rating_mask": rating_mask,
                "box_office_mask": box_office_mask,
                "content_rating_mask": content_rating_mask,
            }

        # PHASE 2 CACHE MODE

        if self.cache_mode == "phase2_cache":
            # Read only a data point's cached frozen outputs upto a
            # certain transformer block.

            assert self.vit_cache is not None
            assert self.bert_cache is not None
            assert self.cache_attention_mask is not None

            vit_embedding = torch.from_numpy(np.array(self.vit_cache[index], copy=True))

            bert_embedding = torch.from_numpy(np.array(self.bert_cache[index], copy=True))

            attention_mask = torch.from_numpy(np.array(self.cache_attention_mask[index], copy=True))

            return {
                "imdb_id": imdb_id,
                "cached_visual_features": vit_embedding,
                "cached_text_features": bert_embedding,
                "attention_mask": attention_mask,
                "features": features,
                "genre_target": genre_target,
                "rating_target": rating_target,
                "box_office_target": box_office_target,
                "content_rating_target": content_rating_target,
                "genre_mask": genre_mask,
                "rating_mask": rating_mask,
                "box_office_mask": box_office_mask,
                "content_rating_mask": content_rating_mask,
            }

        raise RuntimeError(f"Unsupported cache mode: {self.cache_mode}")
