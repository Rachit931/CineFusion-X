from torch.utils.data import DataLoader

from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
)

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

# ViT weights of that particular model

VIT_WEIGHTS = (
    ViT_B_16_Weights.DEFAULT
)

# Preprocessing required for the image to be given input into our ViT

vit_image_transform = VIT_WEIGHTS.transforms()


# Creating BERT's tokenizer by loading it from the pretrained model 

BERT_MODEL = "bert-base-uncased"

# And loads all the preprocessing required for the text to be given input into our BERT

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