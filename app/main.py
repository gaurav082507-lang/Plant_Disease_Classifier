import os
import io
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "PlantVillage.h5"))
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", os.path.join(BASE_DIR, "models", "class_names.json"))
IMG_SIZE = 224   # EfficientNetB0 input size

# ── Custom objects needed to load the model ──────────────────────────────────
# Your model was trained with TF 2.x augmentation layers.
# We override RandomShear to handle the saved config gracefully.
class CompatRandomShear(tf.keras.layers.Layer):
    """Drop-in replacement for RandomShear at inference — just passes input through."""
    def __init__(self, **kwargs):
        kwargs.pop("x_factor", None)
        kwargs.pop("y_factor", None)
        kwargs.pop("fill_mode", None)
        kwargs.pop("interpolation", None)
        kwargs.pop("fill_value", None)
        kwargs.pop("data_format", None)
        super().__init__(**kwargs)

    def call(self, inputs, training=False):
        return inputs

    @classmethod
    def from_config(cls, config):
        return cls(**config)

CUSTOM_OBJECTS = {
    "RandomShear": CompatRandomShear,
}

# ── Load model & class names ─────────────────────────────────────────────────
print("⏳ Loading PlantVillage model...")
model = None
class_names = []

def load_assets():
    global model, class_names
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
        print(f"✅ Model loaded  →  {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Model load failed: {e}")

    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
        print(f"✅ {len(class_names)} classes loaded")
    else:
        print("⚠️  class_names.json not found — falling back to generic labels")
        class_names = [f"Class_{i}" for i in range(38)]

load_assets()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Plant Disease Classifier",
    description="Upload a plant leaf image to detect diseases using EfficientNetB0 trained on PlantVillage.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
        "docs": "/docs",
    }

@app.post("/predict", summary="Predict plant disease from a leaf image")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check models/PlantVillage.h5")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image (jpg/png/webp).")

    contents = await file.read()
    try:
        img_array = preprocess(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    preds = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])

    top5_idx = np.argsort(preds)[::-1][:5]
    top5 = [
        {"class": class_names[i], "confidence": round(float(preds[i]), 4)}
        for i in top5_idx
    ]

    return JSONResponse({
        "filename": file.filename,
        "prediction": class_names[top_idx],
        "confidence": round(confidence, 4),
        "confidence_pct": f"{confidence * 100:.2f}%",
        "top5": top5,
    })

@app.get("/classes", summary="List all 38 plant/disease classes")
def get_classes():
    return {"total": len(class_names), "classes": class_names}

@app.get("/health", summary="Detailed health status")
def health():
    return {
        "api": "ok",
        "model": "loaded" if model is not None else "not loaded",
        "model_path": MODEL_PATH,
        "classes_loaded": len(class_names),
    }
