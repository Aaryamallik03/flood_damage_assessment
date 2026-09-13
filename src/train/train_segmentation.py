"""
Training loop for the flood extent segmentation model (U-Net on SAR imagery).

Callable from a Colab notebook or standalone:
    python -m src.train.train_segmentation --config configs/config.yaml
"""

import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.preprocessing import SARFloodDataset
from src.models.flood_segmentation import FloodSegmentationWrapper
from src.utils.metrics import compute_iou


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    cfg = config["flood_segmentation"]
    data_cfg = config["data"]

    dataset = SARFloodDataset(
        sar_dir=os.path.join(data_cfg["processed_root"], "sar"),
        mask_dir=os.path.join(data_cfg["processed_root"], "masks"),
        patch_size=data_cfg["patch_size"],
    )

    val_size = int(len(dataset) * (1 - data_cfg["train_val_split"]))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    model = FloodSegmentationWrapper(
        encoder=cfg["encoder"],
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    best_val_iou = 0.0

    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0

        for sar, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            sar, mask = sar.to(device), mask.to(device)

            optimizer.zero_grad()
            logits = model(sar)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_iou = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_iou={val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            checkpoint_path = os.path.join(cfg["checkpoint_dir"], "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved to {checkpoint_path}")

    print(f"Training complete. Best val IoU: {best_val_iou:.4f}")


def evaluate(model, dataloader, device):
    model.eval()
    ious = []

    with torch.no_grad():
        for sar, mask in dataloader:
            sar, mask = sar.to(device), mask.to(device)
            pred_mask = model.predict_mask(sar)
            ious.append(compute_iou(pred_mask, mask))

    return sum(ious) / len(ious) if ious else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)
