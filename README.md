# 🌿 Plant Disease Classifier API

A FastAPI-based REST API that detects plant diseases from leaf images using an **EfficientNetB0** model trained on the **PlantVillage** dataset (38 classes).

## 📁 Repo Structure

```
Plant_Disease_Classifier/
├── app/
│   └── main.py              ← FastAPI application
├── models/
│   ├── PlantVillage.h5      ← Trained EfficientNetB0 model (15.6 MB)
│   └── class_names.json     ← 38 PlantVillage class labels
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── test_client.py           ← API test script
```

## 🚀 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API
uvicorn app.main:app --reload

# 3. Open docs
# http://localhost:8000/docs
```

## 🐳 Run with Docker

```bash
docker build -t plant-disease-api .
docker run -p 8000:8000 plant-disease-api
```

## 📡 API Endpoints

| Method | Endpoint    | Description                          |
|--------|-------------|--------------------------------------|
| GET    | `/`         | Health check                         |
| POST   | `/predict`  | Upload leaf image → disease result   |
| GET    | `/classes`  | List all 38 supported classes        |
| GET    | `/health`   | Detailed model status                |
| GET    | `/docs`     | Interactive Swagger UI               |

## 🧪 Test the API

```bash
# Health check
python test_client.py

# With a leaf image
python test_client.py --image path/to/leaf.jpg
```

## 🌱 Supported Plants & Diseases (38 Classes)

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato — covering both healthy and diseased states.
