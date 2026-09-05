import torch
import torch.nn as nn


class MultiModalAttention(nn.Module):
    """
    Multi-head attention for fusing the
    three modalities of the system.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout,
    ):

        super().__init__()

        # Multi head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)

        # Feed-forward network
        # One simple hidden layer
        self.feed_forward = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim * 4,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

        # Layer normalization again after feed forward
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Dropout on Multi-head atttention
        self.dropout = nn.Dropout(dropout)

        # Adding parameters, giving identity(distinctino) to
        # each modality for the cross attention
        self.vis_type = nn.Parameter(torch.randn(1, embedding_dim))
        self.text_type = nn.Parameter(torch.randn(1, embedding_dim))
        self.tab_type = nn.Parameter(torch.randn(1, embedding_dim))

    def forward(self, visual_embedding, textual_embedding, tabular_embedding):
        """
        Inputs :
            visual_embedding:
                [batch_size, 256]

            text_embedding:
                [batch_size, 256]

            tabular_embedding:
                [batch_size, 256]

        Output:
            returned_representation:
                [batch_size, 256]
        """

        # Adding modality(distinction for modality) embedding
        # with the embeddings of the modality itself
        visual_embedding = visual_embedding + self.vis_type
        textual_embedding = textual_embedding + self.text_type
        tabular_embedding = tabular_embedding + self.tab_type

        # Creating the multimodality sequence
        modality_token = torch.stack(
            [visual_embedding, textual_embedding, tabular_embedding],
            dim=1,
        )

        # Shape: [batch size, 3, 256]

        # Multi-head attention
        attention_output, attention_weights = self.attention(
            modality_token, modality_token, modality_token
        )

        # Shape: [batch size, 3, 256]

        # Residual connection + LayerNorm
        x = self.norm1(modality_token + self.dropout(attention_output))

        # Feed-forward network
        feed_forward_ouput = self.feed_forward(x)

        # Second residual connection + LayerNorm
        x = self.norm2(x + self.dropout(feed_forward_ouput))

        # Combining all three modality representation
        fused_representation = x.mean(dim=1)

        # Shape: [batch size, 256]

        return fused_representation
