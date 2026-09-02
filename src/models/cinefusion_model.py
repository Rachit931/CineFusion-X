import torch.nn as nn

from src.models.vit_encoder import ViTEncoder
from src.models.bert_encoder import BERTEncoder
from src.models.mlp_encoder import TabularEncoder
from src.models.self_attention import MultiModalAttention
from src.models.task_heads import Taskheads

class CineFusionModel(nn.Module): 
    """
    Complete Model in Phase 1:
    """

    def __init__(
        self,
        tabular_input_dim,
        tabular_hidden_dim=256,
        embedding_dim=256,
    ):

        super().__init__()

        # Visual Encoder
        self.vit_encoder = ViTEncoder(
            output_dim=embedding_dim,
        )

        # Text Encoder 
        self.bert_encoder = BERTEncoder(
            output_dim=embedding_dim
        )

        # Tabular Encoder 
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            hidden_dim=tabular_hidden_dim,
            output_dim=embedding_dim
        )

        # Multimodel attention
        self.multimodel_attention = MultiModalAttention(
            embedding_dim=embedding_dim,
            num_heads=8,
            dropout=0.1
        )

        # Task Heads
        self.task_heads = Taskheads(
            input_dim=embedding_dim
        )

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        features
    ):
        """
        Inputs: 
            pixel_values:
                Preprocessed poster images:
                [B,3,H,W]
                
            input_ids: 
                Tokenized movie overviews:
                [B, sequence_length]

            attention_mask:
                BERT attention mask:
                [B, sequence_length]

            features:
                Processed tabular features:
                [B, tabular_input_dim]

        Returns:
            Dictionary containing:
                * visual_embedding
                * text_embedding
                * tabular_embedding
                * fused_representation
                * predictions  
        """

        # Visual Branch 
        visual_embedding = self.vit_encoder(
            pixel_values
        )

        # Text Branch 
        text_embedding = self.bert_encoder(
            input_ids,
            attention_mask
        )

        # Tabular branch 
        tabular_embedding = self.tabular_encoder(
            features
        )

        # Multimodal fusion
        fused_representation = self.multimodel_attention(
            visual_embedding,
            text_embedding,
            tabular_embedding
        )

        predictions = self.task_heads(
            fused_representation
        )

        # Return representations and predictions
        return { 
            "visual_embedding": visual_embedding,
            "text_embedding": text_embedding,
            "tabular_embedding": tabular_embedding,
            "fused_representation": fused_representation,
            "predictions": predictions
        }