import torch
import torch.nn as nn

from src.models.bert_encoder import BERTEncoder
from src.models.mlp_encoder import TabularEncoder
from src.models.self_attention import MultiModalAttention
from src.models.task_heads import TaskHeads
from src.models.vit_encoder import ViTEncoder


class MovioModel(nn.Module):
    """
    MOVIO multimodal model.

    Supported modes:

    1. normal
        Raw poster images and tokenized text are passed through the
        complete ViT and BERT encoders.

    2. phase1_cache
        Cached final frozen ViT and BERT backbone representations
        are passed directly to their trainable projection layers.

    3. phase2_cache
        Cached representations from the frozen/trainable boundary
        are passed through the trainable ViT and BERT layers,
        followed by their projection layers.

    The tabular encoder, multimodal attention, and task heads are
    shared across all three modes.
    """

    def __init__(
        self,
        tabular_input_dim: int,
        tabular_hidden_dim: int = 256,
        embedding_dim: int = 256,
        trainable_vit_blocks: int = 0,
        trainable_bert_layers: int = 0,
        tabular_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        cache_mode: str = "normal",
    ):

        super().__init__()

        # Validate cache mode

        valid_cache_modes = {
            "normal",
            "phase1_cache",
            "phase2_cache",
        }

        if cache_mode not in valid_cache_modes:
            raise ValueError(f"cache_mode must be one of {valid_cache_modes}, got '{cache_mode}'")

        self.cache_mode = cache_mode

        # Visual Encoder
        self.vit_encoder = ViTEncoder(
            output_dim=embedding_dim,
            trainable_blocks=trainable_vit_blocks,
        )

        # Text Encoder
        self.bert_encoder = BERTEncoder(
            output_dim=embedding_dim,
            trainable_layers=trainable_bert_layers,
        )

        # Tabular Encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            hidden_dim=tabular_hidden_dim,
            output_dim=embedding_dim,
            dropout=tabular_dropout,
        )

        # Multimodel attention
        self.multimodel_attention = MultiModalAttention(
            embedding_dim=embedding_dim, num_heads=8, dropout=attention_dropout
        )

        # Task Heads
        self.task_heads = TaskHeads(input_dim=embedding_dim)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        cached_visual_embedding: torch.Tensor | None = None,
        cached_text_embedding: torch.Tensor | None = None,
        cached_visual_features: torch.Tensor | None = None,
        cached_text_features: torch.Tensor | None = None,
    ):
        """
        Run MOVIO according to the configured cache mode.

        Inputs:
            features:
                Processed tabular features.
                Shape: [B, tabular_input_dim]

            pixel_values:
                Raw processed poster images.
                Used only in normal mode.
                Shape: [B, 3, 224, 224]

            input_ids:
                Tokenized BERT inputs.
                Used only in normal mode.
                Shape: [B, sequence_length]

            attention_mask:
                BERT attention mask.
                Used in normal mode and Phase 2 cache mode.
                Shape: [B, sequence_length]

            cached_visual_embedding:
                Phase 1 cached ViT backbone representation.
                Shape: [B, 768]

            cached_text_embedding:
                Phase 1 cached BERT backbone representation.
                Shape: [B, 768]

            cached_visual_features:
                Phase 2 cached ViT representation immediately before
                the trainable ViT blocks.
                Shape: [B, num_tokens, 768]

            cached_text_features:
                Phase 2 cached BERT representation immediately before
                the trainable BERT layers.
                Shape: [B, sequence_length, 768]

        Outputs:
            Dictionary containing:
                visual_embedding
                text_embedding
                tabular_embedding
                fused_representation
                predictions
        """

        # Visual and textual branches
        if self.cache_mode == "normal":
            if pixel_values is None:
                raise ValueError("pixel_values is required in normal mode.")

            if input_ids is None:
                raise ValueError("input_ids is required in normal mode.")

            if attention_mask is None:
                raise ValueError("attention_mask is required in normal mode.")

            visual_embedding = self.vit_encoder(pixel_values)

            text_embedding = self.bert_encoder(
                input_ids,
                attention_mask,
            )

        elif self.cache_mode == "phase1_cache":
            if cached_visual_embedding is None:
                raise ValueError("cached_visual_embedding is required in phase1_cache mode.")

            if cached_text_embedding is None:
                raise ValueError("cached_text_embedding is required in phase1_cache mode.")

            visual_embedding = self.vit_encoder.forward_phase1_cached(cached_visual_embedding)

            text_embedding = self.bert_encoder.forward_phase1_cached(cached_text_embedding)

        elif self.cache_mode == "phase2_cache":
            if cached_visual_features is None:
                raise ValueError("cached_visual_features is required in phase2_cache mode.")

            if cached_text_features is None:
                raise ValueError("cached_text_features is required in phase2_cache mode.")

            if attention_mask is None:
                raise ValueError("attention_mask is required in phase2_cache mode.")

            visual_embedding = self.vit_encoder.forward_phase2_cached(cached_visual_features)

            text_embedding = self.bert_encoder.forward_phase2_cached(
                cached_text_features,
                attention_mask,
            )

        else:
            raise RuntimeError(f"Unsupported cache mode: {self.cache_mode}")

        # Tabular

        tabular_embedding = self.tabular_encoder(features)

        # Multimodal fusion

        fused_representation = self.multimodel_attention(
            visual_embedding, text_embedding, tabular_embedding
        )

        # Task heads

        predictions = self.task_heads(fused_representation)

        # Returning

        return {
            "visual_embedding": visual_embedding,
            "text_embedding": text_embedding,
            "tabular_embedding": tabular_embedding,
            "fused_representation": fused_representation,
            "predictions": predictions,
        }
