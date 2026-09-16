"""
FastAPI backend serving the flood damage assessment models.

Run locally (CPU inference is fine - only training needs GPU):
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /health
    POST /predict/damage       - upload pre/post image pair, get damage classification
    POST /predict/flood-extent - upload SAR image, get flood mask
"""

import os

import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.inference.predict import DamagePredictor

app = FastAPI(title="Flood Damage Assessment API", version="0.1.0")

# Allow the dashboard frontend (running on a different port during dev) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before anything beyond local demo use
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "backend/models_store"
damage_predictor = None  # lazy-loaded on first request, or at startup if checkpoint exists

with open("configs/config.yaml") as f:
    _config = yaml.safe_load(f)


@app.on_event("startup")
def load_models():
    global damage_predictor
    checkpoint_path = os.path.join(MODELS_DIR, "damage_classifier.pt")
    if os.path.exists(checkpoint_path):
        damage_predictor = DamagePredictor(
            checkpoint_path=checkpoint_path,
            patch_size=_config["damage_classifier"]["patch_size"],
        )
        print("Damage classifier loaded.")
    else:
        print(f"No checkpoint found at {checkpoint_path} - /predict/damage will error until you train and place one there.")


@app.get("/health")
def health():
    return {"status": "ok", "damage_model_loaded": damage_predictor is not None}


@app.post("/predict/damage")
async def predict_damage(pre_image: UploadFile = File(...), post_image: UploadFile = File(...)):
    if damage_predictor is None:
        return {"error": "Damage classification model not loaded. Train it and place the checkpoint in backend/models_store/."}

    os.makedirs("backend/tmp", exist_ok=True)
    pre_path = f"backend/tmp/{pre_image.filename}"
    post_path = f"backend/tmp/{post_image.filename}"

    with open(pre_path, "wb") as f:
        f.write(await pre_image.read())
    with open(post_path, "wb") as f:
        f.write(await post_image.read())

    result = damage_predictor.predict(pre_path, post_path)
    return result


@app.post("/predict/flood-extent")
async def predict_flood_extent():
    # NOTE: SAR inputs are multi-band GeoTIFFs, not simple uploadable images via
    # a basic file field - wire this up once you have the segmentation checkpoint
    # and decide on the exact input format (raw GeoTIFF upload vs. pre-processed
    # tensor from the pipeline). Left as a stub so the route exists and the API
    # contract is visible to whoever builds the frontend against it.
    return {"error": "Not yet implemented - see comment in backend/main.py"}
