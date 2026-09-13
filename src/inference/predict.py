"""
Inference wrappers - load trained checkpoints and run predictions on new imagery.
Used by both the FastAPI backend and any standalone demo scripts/notebooks.
"""

import torch
from PIL import Image
from torchvision import transforms

from src.models.damage_classifier import CLASS_NAMES, SiameseDamageClassifier
from src.models.flood_segmentation import FloodSegmentationWrapper

_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DamagePredictor:
    def __init__(self, checkpoint_path: str, backbone: str = "resnet50", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SiameseDamageClassifier(backbone=backbone, num_classes=len(CLASS_NAMES), pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device).eval()

    def predict(self, pre_image_path: str, post_image_path: str) -> dict:
        pre_img = _IMAGE_TRANSFORM(Image.open(pre_image_path).convert("RGB")).unsqueeze(0).to(self.device)
        post_img = _IMAGE_TRANSFORM(Image.open(post_image_path).convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(pre_img, post_img)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_class_idx = int(probs.argmax())
        return {
            "damage_class": CLASS_NAMES[pred_class_idx],
            "confidence": float(probs[pred_class_idx]),
            "class_probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
        }


class FloodExtentPredictor:
    def __init__(self, checkpoint_path: str, encoder: str = "resnet34", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = FloodSegmentationWrapper(encoder=encoder, in_channels=2, num_classes=1, pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device).eval()

    def predict(self, sar_tensor: torch.Tensor) -> dict:
        """
        Args:
            sar_tensor: (1, 2, H, W) preprocessed SAR tensor (see preprocessing.SARFloodDataset)
        """
        sar_tensor = sar_tensor.to(self.device)
        mask = self.model.predict_mask(sar_tensor)
        flooded_fraction = mask.mean().item()

        return {
            "flood_mask": mask.cpu().numpy(),
            "flooded_area_fraction": flooded_fraction,
        }
