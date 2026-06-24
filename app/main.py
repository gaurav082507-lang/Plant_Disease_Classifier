import os
import io
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "PlantVillage.h5"))
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", os.path.join(BASE_DIR, "models", "class_names.json"))
IMG_SIZE = 224

# ── Passthrough layer for ALL augmentation layers ─────────────────────────────
# Your model was saved with custom augmentation layers that differ across
# Keras versions. At inference time we just pass the image through unchanged.
class PassThrough(tf.keras.layers.Layer):
    """Identity layer — replaces any augmentation layer at inference."""
    def __init__(self, **kwargs):
        # Drop every kwarg that might cause validation errors
        safe = {k: v for k, v in kwargs.items() if k in ("name", "trainable", "dtype")}
        super().__init__(**safe)

    def call(self, inputs, training=False):
        return inputs

    @classmethod
    def from_config(cls, config):
        return cls(**config)


CUSTOM_OBJECTS = {
    "RandomHeight":   PassThrough,
    "RandomWidth":    PassThrough,
    "RandomFlip":     PassThrough,
    "RandomRotation": PassThrough,
    "RandomZoom":     PassThrough,
    "RandomShear":    PassThrough,
    "RandomContrast": PassThrough,
    "RandomBrightness": PassThrough,
    "RandomTranslation": PassThrough,
    "Rescaling":      PassThrough,
}

# ── Load model & class names ──────────────────────────────────────────────────
print("⏳ Loading PlantVillage model...")
model = None
class_names = []

def load_assets():
    global model, class_names

    # Load class names first (always works)
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
        print(f"✅ {len(class_names)} classes loaded")
    else:
        class_names = [f"Class_{i}" for i in range(38)]
        print("⚠️  class_names.json not found — using generic labels")

    # Try loading model with custom objects
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
        print(f"✅ Model loaded → {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Model load failed: {e}")

load_assets()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Plant Disease Classifier",
    description="Upload a plant leaf image to detect diseases using EfficientNetB0 trained on PlantVillage (38 classes).",
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

@app.get("/health", summary="Detailed health status")
def health():
    return {
        "api": "ok",
        "model": "loaded" if model is not None else "not loaded",
        "classes_loaded": len(class_names),
    }

@app.get("/classes", summary="List all 38 plant/disease classes")
def get_classes():
    return {"total": len(class_names), "classes": class_names}

@app.post("/predict", summary="Predict plant disease from a leaf image")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

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
