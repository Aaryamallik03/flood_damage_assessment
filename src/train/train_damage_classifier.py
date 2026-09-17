"""
Training loop for the Siamese damage classifier.

Callable from a Colab notebook like:
    from src.train.train_damage_classifier import train
    train(config)

Or standalone:
    python -m src.train.train_damage_classifier --config configs/config.yaml

Supports resuming: if checkpoint_dir/last_checkpoint.pt exists, training picks
up from the saved epoch and optimizer state instead of starting over. This
matters a lot on Colab, where a session can disconnect mid-run - without this,
every disconnect meant losing all completed epochs and restarting from zero.
"""

import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.preprocessing import XBDDamageDataset
from src.models.damage_classifier import CLASS_NAMES, SiameseDamageClassifier
from src.utils.metrics import compute_classification_metrics


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    cfg = config["damage_classifier"]
    data_cfg = config["data"]

    dataset = XBDDamageDataset(
        images_dir=os.path.join(data_cfg["xbd_root"], "images"),
        labels_dir=os.path.join(data_cfg["xbd_root"], "labels"),
        patch_size=cfg["patch_size"],
    )

    val_size = int(len(dataset) * (1 - data_cfg["train_val_split"]))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    model = SiameseDamageClassifier(
        backbone=cfg["backbone"],
        num_classes=cfg["num_classes"],
        pretrained=cfg["pretrained"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()  # consider class weights - "destroyed" is rare in most datasets

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    best_val_f1 = 0.0
    start_epoch = 0

    last_checkpoint_path = os.path.join(cfg["checkpoint_dir"], "last_checkpoint.pt")
    if os.path.exists(last_checkpoint_path):
        print(f"Found existing checkpoint at {last_checkpoint_path}, resuming...")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_f1 = checkpoint["best_val_f1"]
        print(f"Resuming from epoch {start_epoch + 1}, best_val_f1 so far: {best_val_f1:.4f}")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        running_loss = 0.0

        for pre_img, post_img, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            pre_img, post_img, labels = pre_img.to(device), post_img.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(pre_img, post_img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, "
              f"val_acc={val_metrics['accuracy']:.4f}, val_f1_macro={val_metrics['f1_macro']:.4f}")

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            checkpoint_path = os.path.join(cfg["checkpoint_dir"], "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved to {checkpoint_path}")

        # Save a full resumable checkpoint after every epoch, not just on improvement -
        # this is what lets a disconnect mid-run cost at most one epoch's progress.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
        }, last_checkpoint_path)

    print(f"Training complete. Best val F1 (macro): {best_val_f1:.4f}")


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for pre_img, post_img, labels in dataloader:
            pre_img, post_img = pre_img.to(device), post_img.to(device)
            outputs = model(pre_img, post_img)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return compute_classification_metrics(all_labels, all_preds, CLASS_NAMES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)
