# Flood Damage Assessment System

AI/ML-based flood damage assessment prototype, developed as a final year project.
Case study: August 26, 2026 Bhotekoshi/Trishuli flash flood (Rasuwa, Nuwakot, Dhading districts, Nepal).

**Scope: working ML prototype**, not a production/official-deployment system. It demonstrates
end-to-end feasibility: flood extent detection + building damage classification + a
dashboard that visualizes results on a map.

---

## Repo Structure

```
flood-damage-assessment/
├── configs/                  # YAML configs for training/inference
├── data/
│   ├── raw/                  # Raw downloaded datasets (xBD, Sentinel scenes) - gitignored
│   └── processed/            # Preprocessed tensors/patches ready for training
├── notebooks/                # Colab notebooks - THIS IS WHERE YOU TRAIN
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_damage_classifier.ipynb
│   ├── 03_train_flood_segmentation.ipynb
│   └── 04_inference_demo.ipynb
├── src/
│   ├── data/                 # Dataset download + preprocessing scripts
│   ├── models/                # Model architectures (Siamese CNN, U-Net)
│   ├── train/                 # Training loop scripts (callable from notebooks or CLI)
│   ├── inference/             # Run trained models on new imagery
│   └── utils/                 # Metrics, visualization helpers
├── backend/                  # FastAPI serving layer (loads trained weights, exposes /predict)
├── frontend/                  # Dashboard (React + Leaflet) - stub for now
├── reports/                   # Case study write-up, results
└── tests/                     # Unit tests for model/data code
```

---

## Why this structure works with Colab-only compute

- All **training** happens in `notebooks/`, run on Colab GPU runtime (T4/A100 depending on plan).
- `src/` holds the actual reusable code — notebooks just `!git clone` this repo (or mount Drive)
  and import from `src/`, so your training logic isn't trapped in unreusable notebook cells.
- Trained model weights (`.pt` files) get saved to Google Drive from Colab, then downloaded into
  `backend/models_store/` for serving/demo purposes.
- The FastAPI backend + dashboard can run locally on your laptop (CPU is fine for *inference*,
  only *training* needs the Colab GPU).

---

## Setup (Colab workflow)

1. Push this repo to GitHub (private repo is fine).
2. In Colab: `!git clone https://github.com/<you>/flood-damage-assessment.git`
3. `%cd flood-damage-assessment && pip install -r requirements.txt`
4. Mount Google Drive to persist datasets/checkpoints across sessions:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
5. Run `notebooks/01_data_exploration.ipynb` first to download and inspect the xBD dataset.
6. Proceed to `02_train_damage_classifier.ipynb`.

See `data/README.md` for dataset acquisition details (xBD, Sentinel-1/2).

---

## Build Order

1. **Damage classification model** (xBD dataset — already labeled, fastest path to a working demo)
2. **Flood extent segmentation** (Sentinel-1 SAR change detection + U-Net)
3. **Backend API** to serve both models
4. **Dashboard** to visualize results on a map
5. **Case study**: run the pipeline on Aug 2026 Nepal flood imagery, compare to reported figures

## License / Attribution
xBD dataset: see https://xview2.org (cite the xView2 paper in your report).
Sentinel data: Copernicus Open Access Hub, free under ESA's data policy.
