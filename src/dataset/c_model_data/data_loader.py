from torch.utils.data import DataLoader

import timm 

from transformers import AutoTokenizer
from src.dataset.c_model_data.custom_dataset import (
    MovieDataset,
)

from config.paths import (
    GENERAL_DIR,
    POSTERS_DIR
)

MASTER_TRAIN = GENERAL_DIR / "master_training.csv"
MASTER_TEST = GENERAL_DIR / "master_test.csv"

# MIM-pretrained ViT-B/16
VIT_MODEL = "vit_base"

vit_model = timm.create_model(
    VIT_MODEL,
    prtrained=True,
    num_classes=0,
)

# Expected prprocessing for image as per the model chosen 
vit_data_config = timm.data.resolve_model_data_config(
    vit_model
)

vit_image_transform = timm.data.create_transform(
    **vit_data_config,
    is_training=False,
)

# Creating BERT's tokenizer by loading it from the pretrained model 
BERT_MODEL = "bert-base-uncased"

# And loads all the processing required for the text to be given input into our BERT
# by applying BERT's tokenizers as processing 
bert_tokenizer = AutoTokenizer.from_pretrained(
    BERT_MODEL
)


# Create train dataset 

train_dataset = MovieDataset(
    MASTER_TRAIN,
    vit_image_transform,
    bert_tokenizer,
    POSTERS_DIR,
    max_text_length=512,
)

test_dataset = MovieDataset(
    MASTER_TEST,
    vit_image_transform,
    bert_tokenizer,
    POSTERS_DIR,
    max_text_length=512,
)

# Create training loader for pytorch 

train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)   

test_loader = DataLoader(
    test_dataset,
    batch_size=20,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)