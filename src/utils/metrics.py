"""Evaluation metrics for both models."""

from typing import List

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report


def compute_classification_metrics(y_true: List[int], y_pred: List[int], class_names: List[str]) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_per_class": dict(zip(class_names, f1_per_class)),
        "report": report,
    }


def compute_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-6) -> float:
    """Intersection over Union for binary segmentation masks."""
    pred_flat = pred_mask.view(-1)
    true_flat = true_mask.view(-1)

    intersection = (pred_flat * true_flat).sum()
    union = pred_flat.sum() + true_flat.sum() - intersection

    return ((intersection + eps) / (union + eps)).item()
