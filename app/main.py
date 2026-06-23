import io
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager

# Global runtime variables
model = None
IMG_HEIGHT = 256
IMG_WIDTH = 256

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles optimized startup logic. Loads the model into RAM once and 
    dynamically inspects configuration to identify target input sizes.
    """
    global model, IMG_HEIGHT, IMG_WIDTH
    try:
        model = tf.keras.models.load_model("models/PlantVillage.h5")
        
        # Dynamically match input resolution to handle 256x256, 224x224, etc.
        input_shape = model.input_shape
        if input_shape and len(input_shape) >= 3:
            IMG_HEIGHT = input_shape[1] if input_shape[1] is not None else 256
            IMG_WIDTH = input_shape[2] if input_shape[2] is not None else 256
        print(f"Model successfully loaded. Input dimensions auto-configured to: ({IMG_HEIGHT}, {IMG_WIDTH})")
    except Exception as e:
        print(f"CRITICAL: Failed to load model weights. Verify path structure. Error: {e}")
    yield
    # Clean up model context cleanly during process shutdown
    if model:
        del model

app = FastAPI(
    title="PlantVillage Disease Classification API",
    description="Production-ready FastAPI setup deploying a Keras H5 vision engine.",
    version="1.0.0",
    lifespan=lifespan
)

# Reference Index mapping for standard PlantVillage implementations
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy", 
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus", "Tomato___tomato_mosaic_virus", "Tomato___healthy"
]

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decodes raw input payloads, maps to RGB, transforms shapes, and rescales values."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img)
        
        # Scale pixel spaces to [0, 1]. Remove division if your layers include explicit Rescaling layers.
        img_array = img_array / 255.0 
        
        # Expand target dims to construct a matching batch element: (1, Height, Width, Channels)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing runtime error: {e}")

@app.get("/health")
def health_check():
    """Service availability heartbeat route."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Classification engine is offline or failed initialization steps.")
        
    # Block incompatible extensions instantly
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid extension format. Use PNG, JPG, or JPEG payloads.")
    
    # Process image streams asynchronously
    image_bytes = await file.read()
    processed_tensor = preprocess_image(image_bytes)
    
    # Execute forward propagation pass
    predictions = model.predict(processed_tensor)
    best_index = np.argmax(predictions[0])
    confidence_score = float(predictions[0][best_index])
    
    # Guard against unexpected shape mismatch arrays securely
    predicted_label = CLASS_NAMES[best_index] if best_index < len(CLASS_NAMES) else f"Unknown Index Class ({best_index})"
    
    return {
        "filename": file.filename,
        "prediction": predicted_label,
        "confidence": round(confidence_score, 4),
        "class_index": int(best_index)
    }
