import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.data.preprocessing import XBDDamageDataset
from src.models.damage_classifier import SiameseDamageClassifier, CLASS_NAMES


IMAGE_DIR = r"data\raw\xbd\images"
LABEL_DIR = r"data\raw\xbd\labels"
CHECKPOINT = "best_model.pt"

SAMPLES_PER_CLASS = 500
BATCH_SIZE = 16


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

dataset = XBDDamageDataset(
    IMAGE_DIR,
    LABEL_DIR
)

# Collect indices for each damage class
class_indices = {
    0: [],
    1: [],
    2: [],
    3: []
}

for index, (_, annotation) in enumerate(dataset.samples):
    label = annotation["damage_label"]
    class_indices[label].append(index)

# Select up to 500 samples from each class
selected_indices = []

for label in range(4):
    indices = class_indices[label][:SAMPLES_PER_CLASS]
    selected_indices.extend(indices)

    print(
        f"{CLASS_NAMES[label]}: "
        f"{len(indices)} samples"
    )

evaluation_dataset = Subset(
    dataset,
    selected_indices
)

loader = DataLoader(
    evaluation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = SiameseDamageClassifier(
    backbone="resnet50",
    num_classes=4,
    pretrained=False
)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=device
    )
)

model.to(device)
model.eval()

predictions = []
true_labels = []

total = len(evaluation_dataset)

print(f"\nEvaluating {total} samples...")

with torch.no_grad():

    processed = 0

    for pre_img, post_img, labels in loader:

        pre_img = pre_img.to(device)
        post_img = post_img.to(device)

        outputs = model(
            pre_img,
            post_img
        )

        preds = torch.argmax(
            outputs,
            dim=1
        ).cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(
            labels.numpy()
        )

        processed += len(labels)

        if processed % 160 == 0:
            print(
                f"Processed {processed}/{total}"
            )


accuracy = accuracy_score(
    true_labels,
    predictions
)

macro_f1 = f1_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 60)
print("STRATIFIED EVALUATION RESULTS")
print("=" * 60)

print(f"Samples evaluated: {total}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1, 2, 3]
    )
)