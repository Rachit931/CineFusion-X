import torch.nn as nn 
import timm 

class ViTEncoder(nn.Module): 
    """
    MAE-pretarined ViT-B/16 encoder 
    """

    def __init__(self, output_dim=256):
        super().__init__()

        # Loading MAE-pretrained ViT-B/16 
        self.vit = timm.create_model(
            "vit_base_patch16_224.mae",
            pretrained=True,
            num_classes=0
        )

        # Model's patches dimensionality reduction
        vit_dim = self.vit.num_features

        self.projection = nn.Linear(
            vit_dim,
            output_dim
        )

    def forward(self, pixel_values): 

        # Patches creation by Model on the input images
        visual_features = self.vit(pixel_values)

        # Creating representations of the input images 
        # and reducing the dimensions of the patches given as input
        visual_embedding = self.projection(
            visual_features
        )

        return visual_embedding