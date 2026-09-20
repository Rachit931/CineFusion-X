from typing import cast

import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class ViTEncoder(nn.Module):
    """
    MAE-pretarined ViT-B/16 encoder.

    Phase 1:
        All ViT blocks are frozen.
        The final ViT backbone output can be cached.
        Only the projection layer remains trainable.

    Phase 2:
        The las N ViT blocks are trainable.
        The output immediately before those trainable blocks
        can be cached.
        The trainable blocks and projection layer remain trainable.
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

        # Model's no. of dimensions for each patch
        vit_dim = cast(int, self.vit.num_features)

        self.projection = nn.Linear(vit_dim, output_dim)

    def forward(self, pixel_values):
        """
        Normal forward pass through the complete ViT.

        User when the cache is not being used.

        Inputs:
            pixel_values:
                    [batch_size,3,224,224]

        Output:
            visual_embedding:
                Shape: [batch_size, output_dim] -> only the CLS token
        """

        if self.trainable_blocks == 0:
            with torch.no_grad():
                visual_features = self.vit(pixel_values)

        else:
            visual_features = self.vit(pixel_values)

        visual_embedding = self.projection(visual_features)

        return visual_embedding

    @torch.no_grad()
    def get_phase1_cache(self, pixel_values):
        """
        Creating the Phase 1 cache.

        All the ViT blocks are frozen, so we caceh teh final
        ViT backbone output before the trainable projection layer.

        Shape:
            [batch_size, 768]
        """

        visual_features = self.vit(pixel_values)

        return visual_features

    def forward_phase1_cached(self, visual_features):
        """
        Use a cached Phase 1 ViT backbone representation.

        The projection layer remains trainable.
        """

        visual_embedding = self.projection(visual_features)

        return visual_embedding

    @torch.no_grad()
    def get_phase2_cache(self, pixel_values):
        """
        Creating the Phase 2 cache.

        The frozen ViT blocks are executed and the touput
        immediately before the trainable ViT blocks is returned.

        For ViT we are using, this representation contains all the
        patch tokens including the CLS token.

        Shape:
            [batch_size, num_tokens, 768]
        """

        if self.trainable_blocks <= 0:
            raise ValueError("Phase 2 cache requires trainable_blocks > 0.")

        vit = self.vit

        # Patch embeddings
        x = vit.patch_embed(pixel_values)

        # Adding positional embeddings and CLS token using timm's
        # own ViT required processing.
        x = vit._pos_embed(x)

        # Apply the same preprocessing used by the ViT before entering
        # the transformer block
        x = vit.patch_drop(x)

        x = vit.norm_pre(x)

        # Number of frozen blocks
        frozen_blocks = self.total_blocks - self.trainable_blocks

        # Running only the trainable transformer blocks
        for block in list(vit.blocks)[:frozen_blocks]:
            x = block[x]

        # All the representations after the frozen layers
        return x

    def forward_phase2_cached(self, cached_features):
        """
        Continue Phase 2 froma  cached frozen representation.

        The cached representation is passed through the trainable
        ViT blocks, followed by the final ViT normalization and
        the trainable projection layer.
        """

        if self.trainable_blocks <= 0:
            raise ValueError("Phase 2 cached forward requires trainable_blocks > 0.")

        x = cached_features

        start_block = self.total_blocks - self.trainable_blocks

        # Only the trainable blocks are executed.
        for block in list(self.vit.blocks)[start_block:]:
            x = block[x]

        # Final normalization used by the ViT after the Transformer blocks.
        x = self.vit.norm(x)

        # Getting only the cls token
        visual_features = x[:, 0]

        visual_embedding = self.projection(visual_features)

        return visual_embedding
