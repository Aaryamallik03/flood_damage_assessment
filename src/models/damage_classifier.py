"""
Siamese CNN for building damage classification.

Takes a pair of image patches (pre-disaster, post-disaster) cropped around a
building footprint, and classifies the damage level.

This mirrors the approach used in the xView2/xBD challenge baselines:
two shared-weight CNN branches process pre/post patches, and their feature
embeddings are concatenated and passed through a classifier head.

Classes (xBD standard):
    0 - no damage
    1 - minor damage
    2 - major damage
    3 - destroyed
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SiameseDamageClassifier(nn.Module):
    def __init__(self, backbone: str = "resnet50", num_classes: int = 4, pretrained: bool = True):
        super().__init__()

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            feature_dim = base_model.fc.in_features
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            feature_dim = base_model.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Drop the final classification layer - we only want the feature extractor.
        # Both branches share the SAME weights (this is what makes it "Siamese").
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_dim = feature_dim

        # Classifier head takes concatenated pre+post embeddings.
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return torch.flatten(feats, 1)

    def forward(self, pre_image: torch.Tensor, post_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pre_image:  (B, 3, H, W) tensor - pre-disaster building patch
            post_image: (B, 3, H, W) tensor - post-disaster building patch
        Returns:
            logits: (B, num_classes)
        """
        pre_feats = self.forward_one(pre_image)
        post_feats = self.forward_one(post_image)
        combined = torch.cat([pre_feats, post_feats], dim=1)
        return self.classifier(combined)


CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


if __name__ == "__main__":
    # Quick sanity check - run this directly to confirm the model builds and
    # produces correctly-shaped output before wiring it into training.
    model = SiameseDamageClassifier(backbone="resnet50", num_classes=4, pretrained=False)
    dummy_pre = torch.randn(2, 3, 224, 224)
    dummy_post = torch.randn(2, 3, 224, 224)
    out = model(dummy_pre, dummy_post)
    print("Output shape:", out.shape)  # expect: torch.Size([2, 4])
