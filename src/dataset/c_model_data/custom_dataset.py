from pathlib import Path 

import pandas as pd 
import torch 
from PIL import Image
from torch.utils.data import Dataset

from config.paths import (
    POSTERS_DIR,
    GENERAL_DIR
)

class MovieDataset(Dataset):

    def __init__(
        self,
        master_path,
        vit_image_preprocessor,
        bert_tokenizer,
        poster_dir,
        max_text_length = 512,
    ): 

        # Load the featurized dataset 
        self.data = pd.read_csv(
            GENERAL_DIR,
            low_memory=False,
        )

        # Store Preprocessors
        self.image_processor = vit_image_preprocessor
        self.tokenizer = bert_tokenizer,
        self.poster_dir = POSTERS_DIR,

        # Features not meant to be used for MLP 
        excluded_columns = [
            "imdb_id",
            "overview",
            "genre_target",
            "rating_target",
            "box_office_target",
            "content_rating_target",
        ]

        # Select only requried preprocessed features
        self.feature_columns = [
            column 
            for column in self.data.columns
            if column not in excluded_columns
        ]
 
        # conversion into tensors
        self.features = torch.tensor(
            self.data[
                self.feature_columns
            ].to_numpy(
                dtype="float32"
            ),dtype = torch.float32,
        )

        # Basic cleaning of overviews
        overview = (
            self.data["overview"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        # TOkenizing all the overview text at once 
        tokenized = bert_tokenizer(
            overview,
            padding="max_length",
            truncation=True,
            max_length = max_text_length,
            return_tensors="pt",
        )

        # Store BERT inputs 
        self.input_ids = tokenized[
            "input_ids"
        ]

        self.attention_mask = tokenized[
            "attention_mask"
        ]

    # Dataset Size

    def __len__(self):

        return len(self.data)

    # Fetching data points as per index

    def __getitem__(self, index): 

        row = self.data.iloc[index]

        # imdb identifier
        imdb_id = row["imdb_id"]

        # matching poster as per the imdb_id
        poster_path = (
            self.poster_dir
            / f"{imdb_id}.jpg"
        )

        if not poster_path.exists(): 
            raise FileNotFoundError(
                f"Poster not found for the {imdb_id}"
                f"{poster_path}"
            )

        # Load Poster 
        image = Image.open(
            poster_path
        ).convert("RGB")

        # Process images for ViT
        pixel_values = self.vit_image_preprocessor(
            image=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        # BERT inputs 
        input_ids = self.input_ids[index]

        attention_mask = self.attention_mask[index]

        # Tabular input 
        features = self.features[index]

        # targets 
        genre_target = row["genre_target"]

        rating_target = row["rating_target"]

        box_office_target = row["box_office_target"]

        content_office_target = row["content_office_target"]

        # Create target masks
        genre_mask = not pd.isna(
            genre_target
        ) 

        box_office_mask = not pd.isna(
            box_office_target
        )

        content_office_mask = not pd.isna(
            content_office_target
        )

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

            "content_office_target": content_office_target,

            "genre_mask": genre_mask,

            "box_office_mask": box_office_mask,

            "content_rating_mask": content_office_mask
        }

    