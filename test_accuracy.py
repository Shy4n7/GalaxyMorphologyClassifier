
import requests
import os

def test_prediction():
    url = 'http://localhost:5001/api/predict'
    # Use a sample image from the frontend folder
    image_path = 'frontend/public/samples/sample_3_Round_Smooth.jpg'
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return

    print(f"Testing prediction on {image_path}...")
    with open(image_path, 'rb') as f:
        files = {'image': f}
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                data = response.json()
                print("\n" + "="*40)
                print(f"PREDICTION RESULT (TTA + NoNorm)")
                print("="*40)
                print(f"Prediction: {data['prediction']}")
                print(f"Confidence: {data['confidence']:.2f}%")
                print("-" * 20)
                print("Individual Guardians:")
                for model, res in data['individual_models'].items():
                    print(f"  {model}: {res['class']} ({res['confidence']:.2f}%)")
                print("="*40 + "\n")
            else:
                print(f"Error: Server returned {response.status_code}")
        except Exception as e:
            print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_prediction()
