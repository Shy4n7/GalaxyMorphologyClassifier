
import requests
import json
import os

# Use a sample image
sample_dir = 'd:/GalaxyClassifier/src/static/samples/random'
if not os.path.exists(sample_dir):
    print("Sample dir not found, checking frontend backup...")
    sample_dir = 'd:/GalaxyClassifier/frontend_backup/public/samples/random'

# Find first jpg
image_path = None
for root, dirs, files in os.walk(sample_dir):
    for f in files:
        if f.endswith('.jpg'):
            image_path = os.path.join(root, f)
            break
    if image_path: break

if not image_path:
    print("No sample image found to test!")
    exit(1)

print(f"Testing with {image_path}...")

try:
    with open(image_path, 'rb') as img:
        files = {'image': img}
        response = requests.post('http://localhost:5001/api/predict', files=files)
        
    if response.status_code == 200:
        data = response.json()
        print("Response received!")
        print(f"Confidence: {data['confidence']}%")
        print(f"Prediction: {data['prediction']}")
        
        models = data.get('individual_models', {})
        print(f"Models returned: {len(models)}")
        for key in models:
            print(f" - {key}: {models[key]['confidence']:.2f}%")
            
        if len(models) != 5:
            print("❌ ERROR: Expected 5 models, got", len(models))
        else:
            print("✅ SUCCESS: All 5 models are present.")
    else:
        print(f"Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Failed to connect: {e}")
