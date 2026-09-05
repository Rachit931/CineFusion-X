import torch.nn as nn


class TabularEncoder(nn.Module):
    """
    MLP as an encoder for the tabular features
    """

    def __init__(self, input_dim, hidden_dim, output_dim):

        super().__init__()

        # MLP
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # Layer 2
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # Layer 3
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # Layer 4
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # Layer 5
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, features):
        """
        Args:
            Input: Preprocessed tabular features
            [ Preprocessed and encoded]

            Output:
                tabular_embddings
                shape: [batch_size, input_dim]

        """

        tabular_embeddings = self.network(features)

        return tabular_embeddings
