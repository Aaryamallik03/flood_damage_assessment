"""
Flood extent segmentation model.

Uses segmentation_models_pytorch's U-Net implementation (well-tested, saves you
from writing U-Net from scratch and debugging it under a deadline).

Input: SAR imagery (Sentinel-1), typically 2 channels (VV + VH polarization).
Output: binary flood mask (1 = flooded, 0 = not flooded).

Why SAR instead of optical: optical satellites (Sentinel-2) are blocked by
monsoon cloud cover, which is exactly when floods happen. SAR penetrates clouds,
so it's the only reliable same-day imagery source after a flood event.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_flood_segmentation_model(
    encoder: str = "resnet34",
    in_channels: int = 2,
    num_classes: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """
    Returns a U-Net model configured for flood segmentation.

    Note: pretrained ImageNet weights expect 3-channel RGB input. Since we're
    feeding 2-channel SAR (VV+VH), smp handles this by adapting the first conv
    layer automatically when in_channels != 3.
    """
    encoder_weights = "imagenet" if pretrained else None

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # raw logits; apply sigmoid/softmax outside during inference
    )
    return model


class FloodSegmentationWrapper(nn.Module):
    """Thin wrapper so this matches the same call pattern as the damage classifier."""

    def __init__(self, encoder: str = "resnet34", in_channels: int = 2, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        self.model = build_flood_segmentation_model(encoder, in_channels, num_classes, pretrained)

    def forward(self, sar_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sar_image: (B, in_channels, H, W) - stacked VV/VH SAR bands
        Returns:
            logits: (B, num_classes, H, W)
        """
        return self.model(sar_image)

    def predict_mask(self, sar_image: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convenience method: returns a binary flood mask instead of raw logits."""
        with torch.no_grad():
            logits = self.forward(sar_image)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()


if __name__ == "__main__":
    model = FloodSegmentationWrapper(encoder="resnet34", in_channels=2, num_classes=1, pretrained=False)
    dummy_sar = torch.randn(2, 2, 256, 256)
    out = model(dummy_sar)
    print("Output shape:", out.shape)  # expect: torch.Size([2, 1, 256, 256])
