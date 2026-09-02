import torch.nn as nn

class TaskHeads(nn.Module):
    """
    Predictions head of the system.
    
    Tasks: 
        * Genre: 19 Label as multi label classification 
        * Ratings: Regression task
        * Box-Office: 4 class multi class classification
        * Content_rating: 4 class multi class classification
    """

    def __init__(self, input_dim): 

        super().__init__()

        # Genre: One logit for each of the 19 genres.
        self.genre_head = nn.Linear(input_dim,19)

        # Ratings: One continuous prediction or logitc
        self.rating_head = nn.Linear(input_dim,1)

        # Box-Office: 
            # 0 = Flop,
            # 1 = Average,
            # 2 = Hit,
            # 3 = Blockbuster,
        self.box_office_head = nn.Linear(input_dim,4)

        # Content-Ratings:
            # 0 = G,
            # 1 = PG,
            # 2 = PG-13,
            # 3 = R,
        self.content_rating_head = nn.Linear(input_dim, 4)

    def forward(self, fused_representation): 
        """
        Inputs: 
            fused_representation: 
                [batch size, input_dim]
        
        Output: 
            Dictionary containing prediction for 
            all four tasks. 
        """

        genre_logits = self.genre_head(fused_representation)

        rating_logit = self.rating_head(fused_representation)

        box_office_logits = self.box_office_head(fused_representation)

        content_rating_logits = self.content_rating_head(fused_representation)

        return {
            "genre": genre_logits,

            "rating": rating_logit,

            "box_office": box_office_logits,

            "content_rating": content_rating_logits,
        }