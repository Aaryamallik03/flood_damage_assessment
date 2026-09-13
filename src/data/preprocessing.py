"""
Preprocessing utilities:
    - filter_flood_events(): narrow xBD to flood-related disasters
    - XBDDamageDataset: PyTorch Dataset for pre/post building patch pairs
    - SARFloodDataset: PyTorch Dataset for SAR image / flood mask pairs
"""

import json
import os
from typing import List, Tuple

import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

FLOOD_DISASTER_KEYWORDS = ["flood", "flooding", "hurricane-harvey", "hurricane-matthew", "midwest-flooding"]

DAMAGE_LABEL_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": 0,  # treat unclassified as no-damage; revisit if this skews results
}


def filter_flood_events(labels_dir: str) -> List[str]:
    """
    Scans xBD label JSONs and returns filenames belonging to flood-related events.
    xBD label files are named like: <event>_<id>_post_disaster.json
    """
    flood_files = []
    for fname in os.listdir(labels_dir):
        if not fname.endswith(".json"):
            continue
        lower = fname.lower()
        if any(keyword in lower for keyword in FLOOD_DISASTER_KEYWORDS):
            flood_files.append(fname)
    return flood_files


class XBDDamageDataset(Dataset):
    """
    Loads (pre_image_patch, post_image_patch, damage_label) triples from xBD.

    Expects xBD's standard structure:
        images/<id>_pre_disaster.png
        images/<id>_post_disaster.png
        labels/<id>_post_disaster.json   (contains building polygons + damage labels)

    For simplicity this crops a fixed-size patch around each building's polygon
    centroid rather than doing precise polygon masking - good enough for a
    classification model, and much simpler to implement under time pressure.
    """

    def __init__(self, images_dir: str, labels_dir: str, patch_size: int = 224, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.patch_size = patch_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.samples = self._build_sample_index()

    def _build_sample_index(self) -> List[Tuple[str, dict]]:
        """Parses label JSONs and returns a flat list of (base_id, building_annotation)."""
        samples = []
        flood_labels = filter_flood_events(self.labels_dir)
        for label_file in flood_labels:
            base_id = label_file.replace("_post_disaster.json", "")
            with open(os.path.join(self.labels_dir, label_file)) as f:
                label_data = json.load(f)
            for feature in label_data.get("features", {}).get("xy", []):
                subtype = feature["properties"].get("subtype", "no-damage")
                samples.append((base_id, {
                    "centroid": self._polygon_centroid(feature["wkt"]),
                    "damage_label": DAMAGE_LABEL_MAP.get(subtype, 0),
                }))
        return samples

    @staticmethod
    def _polygon_centroid(wkt_str: str) -> Tuple[float, float]:
        """Rough centroid extraction from a WKT POLYGON string without a full geometry lib."""
        coords_str = wkt_str.split("((")[1].split("))")[0]
        points = [tuple(map(float, p.strip().split(" "))) for p in coords_str.split(",")]
        xs, ys = zip(*points)
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _crop_patch(self, image: Image.Image, centroid: Tuple[float, float]) -> Image.Image:
        cx, cy = centroid
        half = self.patch_size // 2
        box = (cx - half, cy - half, cx + half, cy + half)
        return image.crop(box)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_id, ann = self.samples[idx]
        pre_path = os.path.join(self.images_dir, f"{base_id}_pre_disaster.png")
        post_path = os.path.join(self.images_dir, f"{base_id}_post_disaster.png")

        pre_img = Image.open(pre_path).convert("RGB")
        post_img = Image.open(post_path).convert("RGB")

        pre_patch = self._crop_patch(pre_img, ann["centroid"])
        post_patch = self._crop_patch(post_img, ann["centroid"])

        pre_tensor = self.transform(pre_patch)
        post_tensor = self.transform(post_patch)
        label = torch.tensor(ann["damage_label"], dtype=torch.long)

        return pre_tensor, post_tensor, label


class SARFloodDataset(Dataset):
    """
    Loads (sar_image, flood_mask) pairs for flood extent segmentation.

    Expects preprocessed GeoTIFFs:
        processed/sar/<id>.tif        - 2-band (VV, VH) SAR image
        processed/masks/<id>.tif      - 1-band binary flood mask
    """

    def __init__(self, sar_dir: str, mask_dir: str, patch_size: int = 256):
        self.sar_dir = sar_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.ids = [f.replace(".tif", "") for f in os.listdir(sar_dir) if f.endswith(".tif")]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        with rasterio.open(os.path.join(self.sar_dir, f"{sample_id}.tif")) as src:
            sar = src.read().astype(np.float32)  # shape: (2, H, W)

        with rasterio.open(os.path.join(self.mask_dir, f"{sample_id}.tif")) as src:
            mask = src.read(1).astype(np.float32)  # shape: (H, W)

        # Normalize SAR backscatter values (dB scale typically ranges roughly -25 to 0)
        sar = np.clip((sar + 25) / 25, 0, 1)

        sar_tensor = torch.from_numpy(sar)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # add channel dim

        return sar_tensor, mask_tensor
