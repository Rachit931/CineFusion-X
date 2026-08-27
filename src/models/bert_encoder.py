import torch.nn as nn 
from transformers import AutoModel

class BERTEncoder(nn.Module):
    """
    BERT encoder takes tokenized 
    movie reviews and returns a text embedding.
    """

    def __init__(self,output_dim = 256):

        super().__init__()

        # Loading the pretrained BERT 
        self.bert = AutoModel.from_pretrained(
            "bert-base-uncased"
        )

        # Tokens number of dimensionality
        bert_dim = self.bert.config.hidden_size

        # Reduction of dimensionality of the tokens
        self.projection = nn.Linear(
            bert_dim,
            output_dim
        )

    def forward(self,input_ids,attention_mask):

        # Passing tokenized inputs into BERT
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        # Take CLS representation
        cls_embedding = outputs.last_hidden_state[:,0,:]

        # Creating representation of the overview texts
        # and reducing the dimensions of the tokens as input
        text_embedding = self.projection(
            cls_embedding
        )

        return text_embedding