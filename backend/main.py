"""
FastAPI backend serving the flood damage assessment models and the website.

Run from the repo root (CPU inference is fine - only training needs GPU):
    pip install fastapi uvicorn python-multipart
    uvicorn backend.main:app --reload --port 8000
Then open http://127.0.0.1:8000

Checkpoint lookup order:
    1. MODEL_PATH environment variable
    2. backend/models_store/damage_classifier.pt
    3. best_model.pt in the repo root
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.inference.predict import DamagePredictor
import mimetypes
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
MODELS_DIR = ROOT / "backend" / "models_store"
HF_REPO_ID = os.environ.get(
    "HF_REPO_ID",
    "Aaryaaaaaa/flood_damage_assessment",
)
HF_FILENAME = os.environ.get(
    "HF_FILENAME",
    "damage_classifier.pt",
)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/tiff": ".tif"}

with open(ROOT / "configs" / "config.yaml") as f:
    _config = yaml.safe_load(f)

damage_predictor = None
load_error = None


def find_checkpoint():
    candidates = []
    if os.environ.get("MODEL_PATH"):
        candidates.append(Path(os.environ["MODEL_PATH"]))
    candidates += [MODELS_DIR / "damage_classifier.pt", ROOT / "best_model.pt"]
    return next((p for p in candidates if p.exists()), None), candidates


@asynccontextmanager
async def lifespan(app: FastAPI):
    global damage_predictor, load_error

    checkpoint, candidates = find_checkpoint()

    if checkpoint is None:
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)

            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                local_dir=str(MODELS_DIR),
            )

            checkpoint = Path(downloaded)
            print(f"Downloaded damage classifier from Hugging Face to {checkpoint}.")

        except Exception as exc:
            load_error = f"Could not download model from Hugging Face: {exc}"
            print(load_error)

    if checkpoint is not None and checkpoint.exists():
        try:
            damage_predictor = DamagePredictor(
                checkpoint_path=str(checkpoint),
                patch_size=_config["damage_classifier"]["patch_size"],
            )
            print(f"Damage classifier loaded from {checkpoint}.")
        except Exception as exc:
            load_error = f"Could not load {checkpoint}: {exc}"
            print(load_error)

    yield

app = FastAPI(title="Flood Damage Assessment API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/privacy")
def privacy():
    return FileResponse(WEB_DIR / "privacy.html")


@app.get("/terms")
def terms():
    return FileResponse(WEB_DIR / "terms.html")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "damage_model_loaded": damage_predictor is not None,
        "error": load_error,
    }


@app.post("/predict/damage")
async def predict_damage(pre_image: UploadFile = File(...), post_image: UploadFile = File(...)):
    if damage_predictor is None:
        raise HTTPException(503, f"Damage model not loaded. {load_error or ''}".strip())
    for upload in (pre_image, post_image):
        if upload.content_type not in SUFFIXES:
            raise HTTPException(400, f"{upload.filename}: upload a PNG, JPEG or TIFF image.")
        if upload.size and upload.size > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Each image must be under 10 MB.")

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for stem, upload in (("pre", pre_image), ("post", post_image)):
            path = Path(tmp) / f"{stem}{SUFFIXES[upload.content_type]}"
            with path.open("wb") as out:
                shutil.copyfileobj(upload.file, out)
            paths.append(str(path))
        try:
            return damage_predictor.predict(paths[0], paths[1])
        except Exception as exc:
            raise HTTPException(500, f"Inference failed: {exc}")


@app.post("/predict/flood-extent")
async def predict_flood_extent():
    # SAR inputs are multi-band GeoTIFFs; wire this up once the segmentation
    # checkpoint and input format are decided. Left as a stub so the route exists.
    raise HTTPException(501, "Not yet implemented - see comment in backend/main.py")