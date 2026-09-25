import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalContrastiveLoss(nn.Module):
    """
    Symmetric visual-text contrastive loss.

    The visual and text embeddings must represent the same batch
    of movies in the same embedding space.

    For a batch of B movies:

        Visual_embeddings : [B,D]
        Text_embeddings   : [B,D]
        Tabular_embeddings: [B,D]

    The similarity matrix is:

        [B,B]

    where:
        similarity[i,i] = positive visual-text pair
        similarity[i,j] = negative pair for i != j

    The loss is calculated in multiple direcitions:

        visual -> text
        text   -> visual

        visual -> tabular
        tabular-> visual

        text   -> tabular
        tabular-> text

    Total contrastive loss :
        (total 6 contrastive losses) / 6
    """

    def __init__(
        self,
        temperature: float = 0.07,
    ):

        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        self.temperature = temperature
        self.contrastive_loss_fn = nn.CrossEntropyLoss()

    def _pairwise_contrastive_loss(
        self,
        anchor_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
    ):
        """
        Calculate one directional contrastive loss.

        Each anchor embedding must identify it's matching target
        embedding among all target embeddings in the batch.

        Input:
            anchor_embeddings: [B,D]
            target_embeddings: [B,D]

        Output: torch.tensor formatted contrastive loss
        alongside with the other previous losses.
        """

        # L2-normalize both modality representations.

        # After normalization, the dot product is used
        # for cosine similarity

        anchor_embeddings = F.normalize(
            anchor_embeddings,
            p=2,
            dim=1,
        )

        target_embeddings = F.normalize(
            target_embeddings,
            p=2,
            dim=1,
        )

        # Compute all anchor-target similarities
        # Transposing the non anchor embeddings
        # [B,D] * [D,B] = [B,B]

        similarity_logits = (
            anchor_embeddings @ target_embeddings.transpose(0, 1)
        ) / self.temperature

        # The correct target for anchor i is target i.
        # Example for B = 4:
        # Labels = [0,1,2,3]
        labels = torch.arange(
            anchor_embeddings.size(0),
            device=anchor_embeddings.device,
        )

        return self.contrastive_loss_fn(similarity_logits, labels)

    def forward(
        self,
        visual_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        tabular_embedding: torch.Tensor,
    ):
        """
        Caculating the six directional contrastive losses.

        Inputs:
            visual_embeddings:
                shape: [B,D]

            text_embeddings:
                shape: [B,D]

            tabular_embeddings:
                shape: [B,D]

        Outputs:
            dict
                {
                "contrastive_loss",
                "visual_to_text_loss",
                "text_to_visual_loss",
                "visual_to_tabular_loss",
                "tabular_to_visual_loss",
                "text_to_tabular_loss",
                "tabular_to_text_loss",
            }
        """

        # Validating the dimensions
        if visual_embedding.dim() != 2:
            raise ValueError("visual_embeddings must have shape [B,D]")

        if text_embedding.dim() != 2:
            raise ValueError("text_embeddings must have shape [B,D]")

        if tabular_embedding.dim() != 2:
            raise ValueError("tabular_embeddings must have shape [B,d]")

        # All three modalities must have the same batch size
        batch_size = visual_embedding.size(0)

        if text_embedding.size(0) != batch_size:
            raise ValueError("visual_embedding and text_embedding must have the same batch size.")

        if tabular_embedding.size(0) != batch_size:
            raise ValueError("visual_embedding and tabular_embedding must have the same batch size")

        # All three embeddings must share the same dimensionality.
        embedding_dim = visual_embedding.size(1)

        if text_embedding.size(1) != embedding_dim:
            raise ValueError(
                "visual_embedding and text_embedding must have the same embedding dimension"
            )

        if tabular_embedding.size(1) != embedding_dim:
            raise ValueError(
                "visual_embedding and tabular_embedding must have the same embedding dimension"
            )

        # Visual <--> Text

        visual_to_text_loss = self._pairwise_contrastive_loss(
            anchor_embeddings=visual_embedding,
            target_embeddings=text_embedding,
        )

        text_to_visual_loss = self._pairwise_contrastive_loss(
            anchor_embeddings=text_embedding,
            target_embeddings=visual_embedding,
        )

        # Visual <--> Tabular

        visual_to_tabular_loss = self._pairwise_contrastive_loss(
            anchor_embeddings=visual_embedding,
            target_embeddings=tabular_embedding,
        )

        tabular_to_visual_loss = self._pairwise_contrastive_loss(
            anchor_embeddings=tabular_embedding,
            target_embeddings=visual_embedding,
        )

        # Text <--> Tabular

        text_to_tabular_loss = self._pairwise_contrastive_loss(
            anchor_embeddings=text_embedding,
            target_embeddings=tabular_embedding,
        )

        tabular_to_text_loss = self._pairwise_contrastive_loss(
            anchor_embeddings=tabular_embedding,
            target_embeddings=text_embedding,
        )

        # Combining all the six directional losses.
        # Equally weighting across all the six directions

        contrastive_loss = (
            visual_to_text_loss
            + text_to_visual_loss
            + visual_to_tabular_loss
            + tabular_to_visual_loss
            + text_to_tabular_loss
            + tabular_to_text_loss
        ) / 6.0

        return {
            "contrastive_loss": contrastive_loss,
            "visual_to_text_loss": visual_to_text_loss,
            "text_to_visual_loss": text_to_visual_loss,
            "visual_to_tabular_loss": visual_to_tabular_loss,
            "tabular_to_visual_loss": tabular_to_visual_loss,
            "text_to_tabular_loss": text_to_tabular_loss,
            "tabular_to_text_loss": tabular_to_text_loss,
        }


class MultiTaskLoss(nn.Module):
    content_rating_class_weights: torch.Tensor
    """
    Calculate the supervised multitask loss of the system.

    Tasks:
        1. Genre          :    Multi-Label Classification
        2. Rating         :    Regression
        3. Box Office     :    4 - class Classification
        4. Content Rating :    4 - class Classification

    Phase 1:
        L_phase1 : L_task

    Phase 2:
        l_phase2 :
            L_task
            + lambda_contrastive * L_contrastive
            + log_variance_contrastive

    TH phase 2 contrastive loss weight becomes a trainable
    log-variance parameter.
    And making it log-varance because it will prevent the
    parameter to become zero while training and will remain
    a positive value throughout.

    Points with missing targets are handled
    during training through the task-specific masks.

    As per the class distribution, Content-Rating classification
    uses class weights to reduce the effectd of class imbalance.
    """

    def __init__(
        self,
        genre_weight: float = 1.0,
        rating_weight: float = 1.0,
        box_office_weight: float = 1.0,
        content_rating_weight: float = 1.0,
        content_rating_class_weights: tuple[float, ...] = (
            1.8,
            0.9,
            0.8,
            0.5,
        ),
        phase: str = "phase1",
        contrastive_temperature: float = 0.07,
        contrastive_weight_init: float = 1.0,
    ):

        super().__init__()

        phase = phase.lower()

        if phase not in {"phase1", "phase2"}:
            raise ValueError("phase must be either 'phase1' or 'phase2'")

        if contrastive_weight_init <= 0:
            raise ValueError("contrastive_weight_init must be greater than 0")

        self.phase = phase

        # Weight of each task in the final loss
        self.genre_weight = genre_weight
        self.rating_weight = rating_weight
        self.box_office_weight = box_office_weight
        self.content_rating_weight = content_rating_weight

        # Validating the content-rating class weights

        if len(content_rating_class_weights) != 4:
            raise ValueError("content_rating_class_weights must contan exact 4 values")

        if any(weight <= 0 for weight in content_rating_class_weights):
            raise ValueError("All content-rating class weights must be greater than 0")

        # Registering these weights as buffer so that they aren't
        # tracked or updated
        self.register_buffer(
            "content_rating_class_weights",
            torch.tensor(
                content_rating_class_weights,
                dtype=torch.float32,
            ),
        )

        # Loss functions
        """
        All these loss functions expects raw logits as input.
        As sigmoid, softmax functions are applied itnernally
        to convert into probabilities to calculate the loss.

        So we have to manually convert the logits into the
        probabilities to get the probabilities for each class
        for that each point.
        """

        # Genre is multi-label classification
        # All 19 genres has an independent binary target.
        self.genre_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        # Rating is a regression problem.
        # Using SmoothL1Loss to make it robust to outliers.
        self.rating_loss_fn = nn.SmoothL1Loss(reduction="none")

        # Box office is currently a 4-class classification target.
        self.box_office_loss_fn = nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=-1,
        )

        # Content rating is a 4-class classification target.
        self.content_rating_loss_fn = nn.CrossEntropyLoss(
            weight=self.content_rating_class_weights,
            reduction="none",
            ignore_index=-1,
        )

        """
        For Multi Class Classification defined loss:
            The encoding of the targets byt the loss
            is done in binary with correct class being as 1
            and others as 0 each point.
            Regardless of assigning any numbers to each class
            in the dataset while preprocessing.
        """

        # PHASE 2 COMPONENTS

        if self.phase == "phase2":
            # Initialize the learnable log-variance

            log_variance_init = -math.log(float(contrastive_weight_init))

            self.log_variance_contrastive = nn.Parameter(
                torch.tensor(
                    log_variance_init,
                    dtype=torch.float32,
                )
            )

            self.contrastive_loss_fn = MultimodalContrastiveLoss(
                temperature=contrastive_temperature
            )

    def _masked_mean(self, loss, mask):
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

        # Making mask a tensor as well
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

        # Making the mask as [B,19] still being 2 dimensional
        # And due to 19 multi label, the loss could become huge
        # so making sure the averaging is adequate because
        # the weight of all four loss is equal.
        mask = mask.expand_as(loss)

        masked_loss = loss * mask

        # Number of points in a batch having valid target.
        # For Genre, each target is independent.
        # So each point's contribution value here equals
        # the no. of labels a point have.

        valid_count = mask.sum()

        # If there are no valid targets for this task
        # in the batch, then making it zero as valid_count will
        # become zero and become invalid.
        if valid_count.item() == 0:
            return loss.sum() * 0.0

        # Returning average loss for that batch.
        return masked_loss.sum() / valid_count

    def _get_contrastive_weight(self):
        """
        Return the positive, learnable Phase-2 contrastive_coefficient,
        """

        if self.phase != "phase2":
            raise RuntimeError("Contrastive weight is only available when phase='phase2'")

        return torch.exp(-self.log_variance_contrastive)

    def forward(
        self,
        predictions,
        targets,
        masks,
        visual_embedding: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        tabular_embedding: torch.Tensor | None = None,
    ):
        """
        Calculate all task losses and the total multitask loss.

        For phase 1 & 2:
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
                Dictionary containing which targets are available
                for each head:
                {
                    "genre": [B],
                    "rating": [B],
                    "box_office": [B],
                    "content_rating": [B]
                }

        For phase2:
            Embeddings:
                Shape: [B,D]

        Returns:
            Phase1:
            Dictionary containing:
                {
                    "total_loss":
                    "genre_loss":
                    "rating_loss":
                    "box_office_loss":
                    "content_rating_loss":
                }
            Phase2:
            Additionally:
                {
                    "contrastive_loss":
                    "lambda_contrastive":
                    "log_variance_contrastive":
                }
                and all six directional contrastive losses.
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
        # Squeezing Cause:
        # the dimensionality of both preidictionn & target
        # should be the same as per the regression loss.
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

        content_rating_loss = self._masked_mean(content_rating_loss, content_rating_mask)

        # COMBINE THE FOUR TASK LOSSES
        supervised_loss = (
            self.genre_weight * genre_loss
            + self.rating_weight * rating_loss
            + self.box_office_weight * box_office_loss
            + self.content_rating_weight * content_rating_loss
        )

        # PHASE 1

        if self.phase == "phase1":
            return {
                "total_loss": supervised_loss,
                "genre_loss": genre_loss,
                "rating_loss": rating_loss,
                "box_office_loss": box_office_loss,
                "content_rating_loss": content_rating_loss,
            }

        # PHASE 2 VALIDATION

        if visual_embedding is None or text_embedding is None or tabular_embedding is None:
            raise ValueError("Phase 2 requires all three embeddings.")

        # CONTRASTIVE LOSS

        contrastive_losses = self.contrastive_loss_fn(
            visual_embedding=visual_embedding,
            text_embedding=text_embedding,
            tabular_embedding=tabular_embedding,
        )

        contrastive_loss = contrastive_losses["contrastive_loss"]

        # LEARNABLE CONTRASTIVE WEIGHT

        lambda_contrastive = self._get_contrastive_weight()

        # PHASE 2 TOTAL OBJECTIVE

        total_loss = (
            supervised_loss + lambda_contrastive * contrastive_loss + self.log_variance_contrastive
        )

        # RETURNING

        return {
            "total_loss": total_loss,
            "genre_loss": genre_loss,
            "rating_loss": rating_loss,
            "box_office_loss": box_office_loss,
            "content_rating_loss": content_rating_loss,
            "contrastive_loss": contrastive_loss,
            "lambda_contrastive": lambda_contrastive,
            "log_variance_contrastive": self.log_variance_contrastive,
        }
