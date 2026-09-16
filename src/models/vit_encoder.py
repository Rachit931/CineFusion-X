from typing import cast

import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class ViTEncoder(nn.Module):
    """
    MAE-pretarined ViT-B/16 encoder
    """

    def __init__(self, output_dim, trainable_blocks=0):
        super().__init__()

        # Loading MAE-pretrained ViT-B/16
        self.vit = cast(
            VisionTransformer,
            timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0),
        )

        # Number of Transformers blocks in ViT
        self.total_blocks = len(self.vit.blocks)

        # Validate configuration
        if trainable_blocks < 0 or trainable_blocks > self.total_blocks:
            raise ValueError(
                f"Trainable blocks must be between 0 and "
                f"{self.total_blocks}. Got {trainable_blocks}"
            )

        self.trainable_blocks = trainable_blocks

        # Freeze the complete ViT backbone first
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze only the last N Trnasformer blocks
        if self.trainable_blocks > 0:
            start_block = self.total_blocks - self.trainable_blocks

            blocks = list(self.vit.blocks)[start_block:]

            for block in blocks:
                for param in block.parameters():
                    param.requires_grad = True

        # Model's patches dimensionality reduction
        vit_dim = cast(int, self.vit.num_features)

        self.projection = nn.Linear(vit_dim, output_dim)

    def forward(self, pixel_values):
        """
        Inputs:
            pixel_values:
                    [batch_size,3,224,224]

        Output:
            visual_embedding:
                Shape: [batch_size, output_dim]
        """

        # Patches creation by Model on the input images

        # Phase 1:
        # Complete ViT backbone is frozen
        # No autograd graph is needed for ViT

        if self.trainable_blocks == 0:
            with torch.no_grad():
                visual_features = self.vit(pixel_values)
        # Phase 2:
        # Some ViT blocks are trainable.
        # Gradients must be tracked and updated
        else:
            visual_features = self.vit(pixel_values)

        # Creating representations of the input images
        # and reducing the dimensions of the patches given as input
        visual_embedding = self.projection(visual_features)

        return visual_embedding
