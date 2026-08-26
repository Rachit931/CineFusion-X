from pathlib import Path 

import pandas as pd 
import torch 
from PIL import Image
from torch.utils.data import Dataset


class MovieDataset(Dataset):

    def __init__(
        self,
        master_path,
        vit_image_transform,
        bert_tokenizer,
        poster_dir,
        max_text_length = 512,
    ): 

        # Load the featurized dataset 
        self.data = pd.read_csv(
            master_path,
            low_memory=False,
        )

        # Store Preprocessors
        self.image_processor = vit_image_transform
        self.tokenizer = bert_tokenizer
        self.poster_dir = Path(poster_dir)

        # finding all genre target columns 
        self.genre_target_columns = [
            column 
            for column in self.data.columns
            if column.startswith("genre_")
            and column.endswith("_target")
        ]

        if len(self.genre_target_columns) != 19:
            raise ValueError(
                f"Expected 19 genre target columns, "
                f"found {len(self.genre_target_columns)}"
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

        # Targets

        self.genre_targets = torch.tensor(
            self.data[
                self.genre_target_columns
            ].to_numpy(
                dtype="float32"
            ),
            dtype=torch.float32,
        )

        self.rating_targets = torch.tensor(
            self.data["rating_target"]
            .fillna(0)
            .to_numpy(
                dtype="float32"
            ),
            dtype=torch.float32,
        )

        self.box_office_targets = torch.tensor(
            self.data["box_office_target"]
            .fillna(-1)
            .to_numpy(
                dtype="int64"
            ),
            dtype=torch.long,
        )

        self.content_rating_targets = torch.tensor(
            self.data["content_rating_target"]
            .fillna(-1)
            .to_numpy(
                dtype="int64"
            ),
            dtype=torch.long,
        )


        # Target masks

        self.genre_masks = (
            self.genre_targets.sum(dim=1) > 0
        )

        self.rating_masks = torch.tensor(
            self.data["rating_target"]
            .notna()
            .to_numpy(),
            dtype=torch.bool,
        )

        self.box_office_masks = torch.tensor(
            self.data["box_office_target"]
            .notna()
            .to_numpy(),
            dtype=torch.bool,
        )

        self.content_rating_masks = torch.tensor(
            self.data["content_rating_target"]
            .notna()
            .to_numpy(),
            dtype=torch.bool,
        )
        
        # Basic cleaning of overviews
        overview = (
            self.data["overview"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        # TOkenizing all the overview text at once 
        tokenized = self.tokenizer(
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
        pixel_values = self.image_processor(
            image
        )

        # BERT inputs 
        input_ids = self.input_ids[index]

        attention_mask = self.attention_mask[index]

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

            "content_rating_mask": content_rating_mask
        }

    