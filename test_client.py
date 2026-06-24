"""
Test client for the Plant Disease Classifier API.

Usage:
    # Test health only
    python test_client.py

    # Test prediction with an image
    python test_client.py --image path/to/leaf.jpg

    # Test against a deployed URL
    python test_client.py --url https://your-deployment-url.com --image leaf.jpg
"""
import argparse
import json
import requests

def test_health(base_url: str):
    print("\n── Health Check ─────────────────────────")
    r = requests.get(f"{base_url}/")
    print(json.dumps(r.json(), indent=2))

def test_classes(base_url: str):
    print("\n── Classes ──────────────────────────────")
    r = requests.get(f"{base_url}/classes")
    data = r.json()
    print(f"Total classes: {data['total']}")
    print("All 38 classes:")
    for i, cls in enumerate(data["classes"]):
        print(f"  {i:2d}. {cls}")

def test_predict(base_url: str, image_path: str):
    print(f"\n── Prediction: {image_path} ──────────────")
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{base_url}/predict",
            files={"file": (image_path, f, "image/jpeg")},
        )
    data = r.json()
    if r.status_code == 200:
        print(f"✅ Prediction : {data['prediction']}")
        print(f"   Confidence : {data['confidence_pct']}")
        print("\nTop 5 predictions:")
        for i, item in enumerate(data["top5"], 1):
            print(f"  {i}. {item['class']}  ({item['confidence']*100:.2f}%)")
    else:
        print(f"❌ Error {r.status_code}: {data}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--image", help="Path to a leaf image for prediction test")
    args = parser.parse_args()

    test_health(args.url)
    test_classes(args.url)
    if args.image:
        test_predict(args.url, args.image)
    else:
        print("\n💡 Tip: Add --image path/to/leaf.jpg to test prediction")
