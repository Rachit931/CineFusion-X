import torch
import torch.nn as nn
from transformers import AutoModel


class BERTEncoder(nn.Module):
    """
    BERT encoder takes tokenized
    movie reviews and returns a text embedding.

    Phase 1:
        Freezing the weights.
        (trainable_layers = 0)
        Only the projection layer remains trainable.
        But all the BERT Transformer layers are frozen.

    Phase 2:
        Not all the weights are frozen.
        (trainable_layers > 0)
        Only th last N BERT Transformers layers are trainable.
        Earlier layres remain froze.
        Project layer still remains trainable.
    """

    def __init__(self, output_dim, trainable_layers=0):

        super().__init__()

        # Loading the pretrained BERT
        self.bert = AutoModel.from_pretrained("bert-base-uncased")

        # Number of Transformer layers in BERT
        self.total_layers = self.bert.config.num_hidden_layers

        # Validating the configuration
        if trainable_layers < 0 or trainable_layers > self.total_layers:
            raise ValueError(
                f"trainable_layers must be between 0 and "
                f"{self.total_layers}. Got {trainable_layers}"
            )

        self.trainable_layers = trainable_layers

        # Freeze the complete BERT backbone
        for param in self.bert.parameters():
            param.require_grad = False

        # Unfreeze only the last N Transformer layers
        if self.trainable_layers > 0:
            start_layer = self.total_layers - trainable_layers

            for layer in self.bert.encoder.layer[start_layer:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Dimensionality of tokens given by our chosen model.
        bert_dim = self.bert.config.hidden_size

        # Reduction of dimensionality of the tokens
        self.projection = nn.Linear(bert_dim, output_dim)

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        """
        Inputs:
            input_ids:
                Tokens IDs from BERT tokenizer.
                Shape: [batch_size, sequence_length]

            attention_mask:
                Distinction between real and padded tokens.
                Shape: [batch_size, sequence_length]

        Returns:
            Text embedding:
                Shape: [batch_size, output_dim]
        """

        # Phase 1:
        # Complete BERT backbon is frozen.
        # No autograd graph is needed for BERT.

        if self.trainable_layers == 0:
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

        # Phase 2:
        # Some BERT layers are trainable.
        # Gradients must be tracked and updated.
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Take CLS representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Creating representation of the overview texts
        # and reducing the dimensions of the tokens as input
        text_embedding = self.projection(cls_embedding)

        return text_embedding
