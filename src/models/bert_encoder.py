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
            param.requires_grad = False

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
        Normal forward pass through the complete BERT model.

        Used when the cache is not being used.

        Inputs:
            input_ids:
                Tokens IDs from BERT tokenizer.
                Shape: [batch_size, sequence_length]

            attention_mask:
                Distinction between real and padded tokens.
                Shape: [batch_size, sequence_length]

        Returns:
            Text embedding:
                Shape: [batch_size, output_dim] -> only the CLS token

        """

        if self.trainable_layers == 0:
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        text_embedding = self.projection(cls_embedding)

        return text_embedding

    @torch.no_grad()
    def get_phase1_cache(self, input_ids, attention_mask):
        """
        Generate the Phase 1 cache.

        All BERT Transformer layers are frozen
        """

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding

    def forward_phase1_cached(self, cache_embedding):
        """
        Use a cached Phase 1 BERT [CLS] representation

        The projection layer still rmains trainable.
        """

        text_embedding = self.projection(cache_embedding)

        return text_embedding

    @torch.no_grad()
    def get_phase2_cache(self, input_ids, attention_mask):
        """
        Generate the Phase 2 cache.

        The frozen BERT layers are executed and the output
        immediately before the trainable BERT layers is
        returned .

        Unlike the Phase 1 cache, this contains the representation
        for every token.

        Shape:
            [batch_size, sequence_lenght, 768]
        """

        if self.trainable_layers <= 0:
            raise ValueError("Phase 2 cache requires trainable_layers > 0")

        bert = self.bert

        # BERT embedding layer
        hidden_states = bert.embeddings(input_ids=input_ids)

        # Build the same extended attention mask used by BERT.
        extended_attention_mask = bert.get_extended_attention_mask(
            attention_mask,
            input_ids.shape,
            input_ids.device,
        )

        # Number of frozen layers
        frozen_layers = self.total_layers - self.trainable_layers

        # Run only the fronzen BERT Layers
        for layer in list(bert.encoder.layer)[:frozen_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
            )[0]

        # This is the exact boundary where the trainable layers begin.
        return hidden_states

    def forward_phase2_cached(self, hidden_states, attention_mask):
        """
        Continue Phase 2 from a cached frozen representation.

        Only the trainable BERT layers will be executed now.
        """

        if self.trainable_layers <= 0:
            raise ValueError("Phase 2 cached forward requires trainable_layers > 0")

        bert = self.bert

        extended_attention_mask = bert.get_extended_attention_mask(
            attention_mask,
            hidden_states.shape[:2],
            hidden_states.device,
        )

        start_layer = self.total_layers - self.trainable_layers

        # Only the trainable layers are executed.
        for layer in list(bert.encoder.layer)[start_layer:]:
            hidden_states = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
            )[0]

        # CLS token
        cls_embedding = hidden_states[:, 0, :]

        text_embedding = self.projection(cls_embedding)

        return text_embedding
