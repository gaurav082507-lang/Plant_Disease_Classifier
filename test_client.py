import requests
import sys
import os

ENDPOINT = "http://127.0.0.1:8000/predict"

def verify_server_pipeline(image_path: str):
    if not os.path.exists(image_path):
        print(f"Execution Error: Missing target verification file target: '{image_path}'")
        return

    print(f"Streaming target binary content [{image_path}] to API...")
    with open(image_path, "rb") as image_file:
        payload = {"file": (os.path.basename(image_path), image_file, "image/png")}
        try:
            response = requests.post(ENDPOINT, files=payload)
            if response.status_code == 200:
                print("\n Inference Route Succeeded! Engine output map:")
                import json
                print(json.dumps(response.json(), indent=4))
            else:
                print(f"Server Error Exception ({response.status_code}): {response.text}")
        except Exception as conn_err:
            print(f"Could not connect to service. Ensure uvicorn runtime is active. Details: {conn_err}")

if __name__ == "__main__":
    # Pull dynamic argument input path string if specified, else look for provided image filenames
    selected_image = sys.argv[1] if len(sys.argv) > 1 else "image_3474a4.png"
    verify_server_pipeline(selected_image)
