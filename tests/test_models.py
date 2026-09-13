"""
Basic sanity tests - confirm model architectures build and produce correctly
shaped output. Run with: pytest tests/
"""

import torch

from src.models.damage_classifier import SiameseDamageClassifier
from src.models.flood_segmentation import FloodSegmentationWrapper


def test_damage_classifier_output_shape():
    model = SiameseDamageClassifier(backbone="resnet18", num_classes=4, pretrained=False)
    pre = torch.randn(2, 3, 224, 224)
    post = torch.randn(2, 3, 224, 224)
    out = model(pre, post)
    assert out.shape == (2, 4)


def test_flood_segmentation_output_shape():
    model = FloodSegmentationWrapper(encoder="resnet18", in_channels=2, num_classes=1, pretrained=False)
    sar = torch.randn(2, 2, 256, 256)
    out = model(sar)
    assert out.shape == (2, 1, 256, 256)
