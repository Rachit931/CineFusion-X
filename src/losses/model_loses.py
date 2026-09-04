import torch 
import torch.nn as nn 

class MultiTaskLoss(nn.Module):
    """
    Calculate the supervised multitask loss of the system.
    
    Tasks: 
        1. Genre          :    Multi-Label Classification
        2. Rating         :    Regression
        3. Box Office     :    4 - class Classification
        4. Content Rating :    4 - class Classification
    
    Points with missing targets are handled 
    during training through the task-specific masks.
    """

    def __init__(
        self,
        genre_weight = 1.0,
        rating_weight = 1.0,
        box_office_weight = 1.0,
        content_rating_weight = 1.0,
    ):

        super().__init__()

        # Weight of each task in the final loss 
        self.genre_weight = genre_weight
        self.rating_weight = rating_weight
        self.box_office_weight = box_office_weight
        self.content_rating_weight = content_rating_weight

        # Loss functions 

        # Genre is multi-label classification 
        # All 19 genres has an independent binary target. 
        self.genre_loss_fn = nn.BCEWithLogitsLoss(
            reduction="none"
        )

        # Rating is a regression problem.
        # Using SmoothL1Loss to make it robust to outliers. 
        self.rationg_loss_fn = nn.SmoothL1Loss(
            reduction="none"
        )

        # Box office is currently a 4-class classification target. 
        self.box_office_loss_fn = nn.CrossEntropyLoss(
            reduction="none"
        )

        # Content rating is a 4-class classification target. 
        self.content_rating_loss_fn = nn.CrossEntropyLoss(
            reduction="none"
        )

    def _masked_mean(self,loss,mask): 
        """
        Calculates the mean loss only over valid targets. 
        
        Inputs: 
            loss: 
                Loss values before masking.
            
            mask: 
                1 / True   :   Target is available
                0 / False  :   Target is missing
                
        Output: 
            Scalar masked values. 
        """

        #Making mask a tensor as well 
        mask = mask.to(dtype=loss.dtype)

        # Broadcasting the mask over the loss 


        # As of now, for genre, 
        # Loss is [B,19] : 2 dimensional 
        # And mask is [B] : 1 dimensional 
        # Unsqueezing the mask to make it 2 dimensional 
        # For other 3 heads, this loop will be skipped as their 
        # loss is 1 dimensional itself already.  

        while mask.dim() < loss.dim(): 
            mask = mask.unsqueeze(-1)

        masked_loss = loss * mask 

        valid_count = mask.sum()

        # If there are no valid targets for this task 
        # in the batch, then making it zero as valid_count will 
        # become zero and become invalid.
        if valid_count.item() == 0: 
            return loss.sum() * 0.0

        return masked_loss.sum() / valid_count

    def forward(
        self,
        predictions,
        targets,
        masks,
    ):
        """
        Calculate all task losses and the total multitask loss.
        
        Input:
            predictions: 
                Obtained from task_heads.
                Dictionary containing model predictions for each head:
                {
                    "genre": [B,19],
                    "rating": [B,1],
                    "box_office: [B,4],
                    "content_rating": [B,4]
                }
            targets:
                Obtained from custom_dataset.
                Dictionary containing actual targets for each head:
                {
                    "genre": [B,19],
                    "rating": [B],
                    "box_office": [B],
                    "content_rating": [B]
                }
            masks:
                Obtained from custom_dataset.
                Dictionary containing which targets are availabe 
                for each head: 
                {
                    "genre": [B],
                    "rating": [B],
                    "box_office": [B],
                    "content_rating": [B]
                }
        
        Returns:
            Dictionary containing: 
                {
                    "total_loss": 
                    "genre_loss":
                    "rating_loss":
                    "box_office_loss":
                    "content_rating_loss":
                }
        """

        # GENRE LOSS 
        genre_predictions = predictions["genre"]
        genre_targets = targets["genre"]
        genre_mask = masks["genre"]

        genre_loss = self.genre_loss_fn(
            genre_predictions,
            genre_targets.float(),
        )

        genre_loss = self._masked_mean(
            genre_loss,
            genre_mask,
        )

        # RATING LOSS 
        # Squeezing the predictn as the dimensionality of both 
        # preidictn & target should be the same as per the regression loss.
        rating_predictions = predictions["rating"].squeeze(-1)
        rating_targets = targets["rating"].float()
        rating_mask = masks["rating"]

        rating_loss = self.rating_loss_fn(
            rating_predictions,
            rating_targets,
        )

        rating_loss = self._masked_mean(
            rating_loss,
            rating_mask,
        )

        # BOX-OFFICE LOSS 
        box_office_predictions = predictions["box_office"]
        box_office_targets = targets["box_office"].long()
        box_office_mask = masks["box_office"]

        box_office_loss = self.box_office_loss_fn(
            box_office_predictions,
            box_office_targets,
        )

        box_office_loss = self._masked_mean(
            box_office_loss,
            box_office_mask,
        )

        # CONTENT-RATING LOSS 
        content_rating_predictions = predictions["content_rating"]
        content_rating_targets = targets["content_rating"].long()
        content_rating_mask = masks["content_rating"]

        content_rating_loss = self.content_rating_loss_fn(
            content_rating_predictions,
            content_rating_targets,
        )

        content_rating_loss = self._masked_mean(
            content_rating_loss,
            content_rating_mask
        )

        # COMBINE THE FOUR TASK LOSSES 
        total_loss = (
            self.genre_weight * genre_loss
            + self.rating_weight * rating_loss
            + self.box_office_weight * box_office_loss
            + self.content_rating_weight * content_rating_loss
        )

        return {
            "total_loss": total_loss,
            "genre_loss": genre_loss,
            "rating_loss": rating_loss,
            "box_office_loss": box_office_loss,
            "content_rating_loss": content_rating_loss,
        }
    