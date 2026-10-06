# ðŸŒŠ Flood Damage Assessment Using Deep Learning

**Live demo:** https://aaryamallik03.github.io/flood_damage_assessment/ (runs in your browser, nothing is uploaded)


An end-to-end deep learning project for assessing disaster impact from satellite and aerial imagery. The system combines **building-level damage classification** from pre- and post-disaster imagery with **flood extent segmentation** using SAR data.

The project is designed to support rapid post-disaster assessment by automatically identifying damaged buildings and mapping flooded areas.

---

## ðŸš€ Project Overview

Natural disasters such as floods can damage buildings and infrastructure across large geographic areas. Manual assessment is time-consuming and difficult, especially when affected regions are difficult to access.

This project explores how deep learning and remote-sensing imagery can automate parts of the disaster assessment process.

### The system contains two major components:

| Component                         | Input                       | Output            | Approach                |
| --------------------------------- | --------------------------- | ----------------- | ----------------------- |
| ðŸ  Building Damage Classification | Pre- & post-disaster images | Damage severity   | Siamese CNN + ResNet-50 |
| ðŸŒŠ Flood Extent Segmentation      | SAR imagery                 | Flooded-area mask | Semantic segmentation   |

### Building damage classes

The damage classifier predicts four xBD-style categories:

* **No Damage**
* **Minor Damage**
* **Major Damage**
* **Destroyed**

---

## ðŸ§  System Architecture

### 1. Building Damage Classification

The damage classifier uses a **Siamese convolutional neural network** with shared ResNet-50 encoders.

Two corresponding image patches are processed:

```text
             Pre-disaster image
                    â”‚
                    â–¼
              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
              â”‚ ResNet-50 â”‚
              â””â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜
                    â”‚
                    â–¼
              Feature Vector
                    â”‚
                    â”‚
                    â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                    â”‚              â”‚
                    â–¼              â–¼
              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
              â”‚ Feature Concatenation  â”‚
              â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â”‚
                           â–¼
                    Classification
                           â”‚
                           â–¼
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚ No / Minor / Major / Destroyed â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â–²
                           â”‚
                    Feature Vector
                           â–²
                           â”‚
              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
              â”‚ ResNet-50  â”‚
              â””â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜
                    â”‚
                    â–¼
             Post-disaster image
```

The two branches share the same encoder weights, allowing the network to learn representations from both pre- and post-disaster imagery before comparing their combined features.

---

## ðŸŒŠ 2. Flood Extent Segmentation

The second component works with **Synthetic Aperture Radar (SAR)** imagery.

Unlike optical imagery, SAR can be useful in conditions where clouds or poor visibility make conventional imagery difficult to use.

The segmentation pipeline processes SAR data and produces a pixel-level flood mask:

```text
SAR Image
    â”‚
    â–¼
Preprocessing
    â”‚
    â–¼
Segmentation Model
    â”‚
    â–¼
Flood Probability / Mask
    â”‚
    â–¼
Flooded Area Map
```

---

## ðŸ“Š Dataset

The building damage component uses the **xBD/xView2 disaster-damage dataset**.

The preprocessing pipeline filters the available disaster labels for flood-related events and extracts building-centered image patches from the corresponding pre- and post-disaster imagery.

### Building samples used

**45,709 building samples**

Class distribution:

| Damage Class |    Samples |
| ------------ | ---------: |
| No Damage    |     23,424 |
| Minor Damage |      9,360 |
| Major Damage |     10,302 |
| Destroyed    |      2,623 |
| **Total**    | **45,709** |

The dataset is imbalanced, with destroyed buildings representing the smallest class.

---

## âš™ï¸ Preprocessing

For each building annotation:

1. The building footprint is read from the disaster label.
2. The building centroid is calculated.
3. A corresponding patch is extracted from the pre-disaster image.
4. The same spatial region is extracted from the post-disaster image.
5. Images are resized to **224 Ã— 224**.
6. Images are converted to tensors.
7. ImageNet normalization is applied.

The same spatial location is therefore compared before and after the disaster.

---

## ðŸ—ï¸ Model

### Siamese ResNet-50

The damage classification model consists of:

* ResNet-50 image encoder
* Shared weights between pre- and post-disaster branches
* Feature concatenation
* Fully connected classification head
* ReLU activations
* Dropout regularization
* Four-class output

### Classification head

```text
2048 + 2048 features
        â”‚
        â–¼
     Linear
      4096 â†’ 512
        â”‚
       ReLU
        â”‚
     Dropout
        â”‚
        â–¼
     Linear
       512 â†’ 128
        â”‚
       ReLU
        â”‚
     Dropout
        â”‚
        â–¼
     Linear
       128 â†’ 4
        â”‚
        â–¼
Damage Class
```

---

## ðŸ“ˆ Model Evaluation

A **2,000-sample stratified diagnostic evaluation** was performed using 500 samples from each damage category.

> **Important:** this is a diagnostic evaluation subset drawn from the overall dataset, not a clean independent test set. Because the original training process used a random split without saving the split indices, these results should not be interpreted as final generalization performance.

### Diagnostic results

| Metric            |     Result |
| ----------------- | ---------: |
| Samples evaluated |      2,000 |
| Accuracy          | **96.45%** |
| Macro F1          | **96.43%** |

### Per-class performance

| Class        | Precision | Recall |   F1 |
| ------------ | --------: | -----: | ---: |
| No Damage    |      0.94 |   1.00 | 0.97 |
| Minor Damage |      0.95 |   0.99 | 0.97 |
| Major Damage |      0.98 |   0.96 | 0.97 |
| Destroyed    |      1.00 |   0.90 | 0.95 |

### Confusion Matrix

```text
                 Predicted
              No   Min  Maj  Des

Actual No     499    1    0    0
       Min      3  496    1    0
       Maj     10    8  482    0
       Des     21   18    9  452
```

The most notable error pattern is confusion involving the **destroyed** class, which has the smallest number of training samples.

---

## ðŸ§ª Example Inference

The trained checkpoint can be loaded with the inference pipeline:

```python
from src.inference.predict import DamagePredictor

predictor = DamagePredictor("best_model.pt")

result = predictor.predict(
    "path/to/pre_disaster.png",
    "path/to/post_disaster.png"
)

print(result)
```

Example output:

```text
{
    'damage_class': 'no-damage',
    'confidence': 0.9997,
    'class_probabilities': {
        'no-damage': 0.9997,
        'minor-damage': 0.0002,
        'major-damage': 0.00005,
        'destroyed': 0.00002
    }
}
```

> **Inference note:** the current `DamagePredictor` interface expects image paths, but the model was trained on building-centered patches. For production-quality building-level inference, the inference pipeline should first identify or receive a building footprint/centroid and crop the corresponding patch before classification.

---

## ðŸ—‚ï¸ Project Structure

```text
flood-damage-assessment/
â”‚
â”œâ”€â”€ backend/
â”‚   â””â”€â”€ main.py
â”‚
â”œâ”€â”€ configs/
â”‚   â””â”€â”€ config.yaml
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data/
â”‚   â”‚   â”œâ”€â”€ bipad_client.py
â”‚   â”‚   â”œâ”€â”€ download_sentinel*.py
â”‚   â”‚   â”œâ”€â”€ download_xbd.py
â”‚   â”‚   â”œâ”€â”€ ndrrma_sitre*.py
â”‚   â”‚   â””â”€â”€ preprocessing.py
â”‚   â”‚
â”‚   â”œâ”€â”€ inference/
â”‚   â”‚   â””â”€â”€ predict.py
â”‚   â”‚
â”‚   â”œâ”€â”€ models/
â”‚   â”‚   â”œâ”€â”€ damage_classifier.py
â”‚   â”‚   â””â”€â”€ flood_segmentation.py
â”‚   â”‚
â”‚   â”œâ”€â”€ train/
â”‚   â”‚   â”œâ”€â”€ train_damage_classifier.py
â”‚   â”‚   â””â”€â”€ train_segmentation*.py
â”‚   â”‚
â”‚   â””â”€â”€ utils/
â”‚       â””â”€â”€ metrics.py
â”‚
â”œâ”€â”€ tests/
â”‚
â”œâ”€â”€ evaluate_model.py
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

---

## ðŸ› ï¸ Tech Stack

**Programming**

* Python
* PyTorch
* Torchvision
* NumPy
* OpenCV
* Rasterio
* scikit-learn

**Deep Learning**

* ResNet-50
* Siamese CNN architecture
* Semantic segmentation
* Transfer learning

**Remote Sensing**

* xBD/xView2 imagery
* SAR imagery
* Pre/post-disaster image analysis
* Flood-mask generation

**Development**

* Google Colab
* Git
* GitHub

---

## ðŸ’» Installation

Clone the repository:

```bash
git clone https://github.com/Aaryamallik03/flood_damage_assessment.git
cd flood_damage_assessment
```

Create a virtual environment:

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## â–¶ï¸ Running the Project

### Train the damage classifier

```bash
python src/train/train_damage_classifier.py
```

### Evaluate the model

```bash
python evaluate_model.py
```

### Run inference

```python
from src.inference.predict import DamagePredictor

predictor = DamagePredictor("best_model.pt")

result = predictor.predict(
    "pre_disaster.png",
    "post_disaster.png"
)

print(result)
```

---

## ðŸ“¦ Model Checkpoint

The trained `best_model.pt` checkpoint is approximately **98 MB** and is intentionally not stored directly in the Git repository.

The repository contains the model architecture and inference code required to load the checkpoint.

For reproducibility, the trained checkpoint can be distributed separately or through Git LFS / release assets.

---

## ðŸ”¬ Current Limitations

This project is still under development.

Important limitations include:

* The current damage evaluation is not based on an independent event-level test set.
* The original training script uses a random train/validation split.
* Related buildings from the same disaster imagery can therefore potentially appear across different splits.
* The current inference wrapper needs building-centered cropping to exactly match the training setup.
* The segmentation component requires further evaluation on a properly held-out SAR dataset.
* Real-world deployment would require additional geographic and disaster-event validation.

---

## ðŸš§ Future Improvements

Planned improvements include:

* [ ] Event-level train/validation/test splitting
* [ ] Independent disaster-event evaluation
* [ ] Building footprint detection for automatic cropping
* [ ] Improved handling of class imbalance
* [ ] Precision/recall and calibration analysis
* [ ] SAR flood segmentation evaluation
* [ ] Interactive disaster-assessment dashboard
* [ ] Map-based visualization of damaged buildings and flooded areas
* [ ] REST API for model inference
* [ ] Deployment on cloud infrastructure
* [ ] Explainable AI using Grad-CAM or related visualization techniques

---

## ðŸŽ¯ Project Goal

The long-term goal is to develop a practical **AI-assisted disaster assessment pipeline** capable of combining multiple sources of remote-sensing data to provide rapid information about:

**Where is the flooding?**

**Which buildings are damaged?**

**How severe is the damage?**

This can help demonstrate how computer vision, deep learning, and geospatial data can be combined for real-world disaster-response applications.

---

## ðŸ‘©â€ðŸ’» Author

**Aarya Mallik**

B.Tech Computer Science Engineering
NIT Meghalaya

Interested in **Data Science, Machine Learning, Computer Vision, and AI for real-world applications**.

---

## â­ Acknowledgements

This project builds upon publicly available disaster-imagery and remote-sensing datasets and open-source deep learning tools.

If you find the project useful, consider â­ starring the repository.
