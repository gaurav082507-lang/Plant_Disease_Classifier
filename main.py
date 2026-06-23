import io
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

app = FastAPI(
    title="PlantVillage Disease Classification API",
    description="An API to classify plant diseases from leaf images using a Keras H5 model."
)

# 1. Load the trained H5 model
try:
    model = tf.keras.models.load_model("PlantVillage.h5")
except Exception as e:
    raise RuntimeError(f"Could not load model 'PlantVillage.h5'. Verify the file path. Error: {e}")

# 2. Configuration (Adjust dimensions based on your specific model architecture)
IMG_SIZE = (256, 256) 

# Standard PlantVillage 38-class reference.
# Rearrange or shorten this list to perfectly match the exact indices of your training generator classes.
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
    """Decodes, resizes, and normalizes incoming raw image bytes."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)
        img_array = np.array(image)
        
        # Scale pixel values to [0, 1]. Remove this if your model already contains a Rescaling layer.
        img_array = img_array / 255.0 
        
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

@app.get("/")
def root():
    return {
        "status": "Online",
        "message": "PlantVillage Classification API is running. Go to /docs to test the endpoints."
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate payload extension
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a JPG, JPEG, or PNG image.")
    
    # Read raw content
    image_bytes = await file.read()
    
    # Preprocess image data
    processed_image = preprocess_image(image_bytes)
    
    # Generate predictions
    predictions = model.predict(processed_image)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    # Map index safely to class name string
    predicted_label = (
        CLASS_NAMES[predicted_class_idx] 
        if predicted_class_idx < len(CLASS_NAMES) 
        else f"Unknown Class Index ({predicted_class_idx})"
    )
    
    return {
        "filename": file.filename,
        "prediction": predicted_label,
        "confidence": round(confidence, 4),
        "class_index": int(predicted_class_idx)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
