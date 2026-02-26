"""
Galaxy Classifier Inference Server
Web app for classifying galaxy images using trained ensemble models
"""

import os
import sys
import io
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torchvision.transforms as transforms
import torchvision.models as models
import base64
import cv2

# Add src to path
sys.path.append(os.path.dirname(__file__))
from gradcam_utils import generate_gradcam, generate_ensemble_gradcam, create_gradcam_grid

app = Flask(__name__, static_folder='static')
CORS(app)

# Galaxy class names
CLASS_NAMES = [
    "Disturbed Galaxies",
    "Merging Galaxies", 
    "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies",
    "Cigar Shaped Smooth Galaxies",
    "Barred Spiral Galaxies",
    "Unbarred Tight Spiral Galaxies",
    "Unbarred Loose Spiral Galaxies",
    "Edge-on Galaxies without Bulge",
    "Edge-on Galaxies with Bulge"
]

# Model architectures (same as training script)
class OverfitModel(nn.Module):
    """Model architecture matching the training script for overfitted models"""
    def __init__(self, model_type='efficientnet', num_classes=10):
        super(OverfitModel, self).__init__()
        
        if model_type == 'resnet50':
            self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            in_features = self.base.fc.in_features
            self.base.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        elif model_type == 'densenet121':
            self.base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            in_features = self.base.classifier.in_features
            self.base.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        elif model_type == 'mobilenet':
            self.base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            in_features = self.base.classifier[0].in_features
            self.base.classifier = nn.Sequential(
                nn.Linear(in_features, 1280),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, num_classes)
            )
        else:  # efficientnet
            self.base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = self.base.classifier[1].in_features
            self.base.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        return self.base(x)

# Global models dictionary
MODELS = {}
# DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_best_model(model_type):
    """Find the highest accuracy overfit model in models/ directory"""
    models_dir = 'models'
    best_acc = 0.0
    best_path = None
    
    if not os.path.exists(models_dir):
        return None, 0.0
    
    # Pattern: {model_type}_overfit_acc{acc}.pth or {model_type}_overfit_hard_acc{acc}.pth
    for filename in os.listdir(models_dir):
        # Check for both standard and hard mode patterns
        if filename.startswith(f"{model_type}_overfit_") and "acc" in filename and filename.endswith(".pth"):
            try:
                acc_str = filename.split('acc')[-1].replace('.pth', '')
                acc = float(acc_str)
                if acc > best_acc:
                    best_acc = acc
                    best_path = os.path.join(models_dir, filename)
            except:
                continue
                
    return best_path, best_acc

def load_models():
    """Load the best overfitted models available"""
    global MODELS
    
    print(f"Loading models on {DEVICE}...")
    
    # Configuration: (name, model_type)
    configs = [
        ('resnet50', 'resnet50'),
        ('densenet121', 'densenet121'),
        ('efficientnet_b0_v1', 'efficientnet'),
        ('efficientnet_b0_v2', 'efficientnet'),
        ('mobilenet_v3', 'mobilenet')
    ]
    
    loaded_count = 0
    
    for name, model_type in configs:
        path, acc = find_best_model(model_type)
        
        if path:
            print(f"Found {name}: {os.path.basename(path)} (Acc: {acc:.2f}%)")
            try:
                model = OverfitModel(model_type=model_type, num_classes=10)
                state_dict = torch.load(path, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                model.eval()
                MODELS[name] = model
                loaded_count += 1
                print(f"✓ Loaded {name}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
        else:
            print(f"✗ No overfit model found for {model_type}")
            
    print(f"Successfully loaded {loaded_count} models")

def preprocess_image(image_bytes):
    """Preprocess uploaded image for model inference with TTA"""
    # Open image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize to 69x69 (same as training)
    # Note: No normalization because training data was [0, 1] raw pixels
    transform = transforms.Compose([
        transforms.Resize((69, 69)),
        transforms.ToTensor()
    ])
    
    # Original image tensor (1, 3, 69, 69)
    x = transform(img).unsqueeze(0).to(DEVICE)
    
    # Test-Time Augmentation (TTA)
    # Generate 4 variants: 0, 90, 180, 270 degrees rotation
    # Galaxy morphology is rotation invariant, so this boosts robustness significantly
    x90 = torch.rot90(x, 1, [2, 3])
    x180 = torch.rot90(x, 2, [2, 3])
    x270 = torch.rot90(x, 3, [2, 3])
    
    # Batch of 4 variants
    batch = torch.cat([x, x90, x180, x270], dim=0)
    return batch

def predict_ensemble(image_tensor):
    """Run ensemble prediction with all models using TTA averaging"""
    if not MODELS:
        raise RuntimeError("No models loaded")
    
    predictions = []
    model_results = {}
    
    with torch.no_grad():
        for name, model in MODELS.items():
            # Pass TTA batch (4, 3, 69, 69)
            output = model(image_tensor)
            
            # Get probabilities for each variant (4, 10)
            probs_batch = torch.softmax(output, dim=1).cpu().numpy()
            
            # Average probabilities across TTA variants (Robustness boost!)
            probs = np.mean(probs_batch, axis=0)
            
            predictions.append(probs)
            
            # Store individual model prediction
            pred_class = np.argmax(probs)
            model_results[name] = {
                'class': CLASS_NAMES[pred_class],
                'confidence': float(probs[pred_class] * 100)
            }
    
    # Ensemble averaging across models
    ensemble_probs = np.mean(predictions, axis=0)
    pred_class = np.argmax(ensemble_probs)
    confidence = ensemble_probs[pred_class] * 100
    
    # Get top 3 predictions
    top3_indices = np.argsort(ensemble_probs)[-3:][::-1]
    top3_predictions = [
        {
            'class': CLASS_NAMES[i],
            'confidence': float(ensemble_probs[i] * 100)
        }
        for i in top3_indices
    ]
    
    return {
        'prediction': CLASS_NAMES[pred_class],
        'confidence': float(confidence),
        'top3': top3_predictions,
        'individual_models': model_results,
        'all_probabilities': {CLASS_NAMES[i]: float(ensemble_probs[i] * 100) for i in range(10)}
    }

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/classifier')
def classifier():
    return send_from_directory(app.static_folder, 'classifier.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        image_tensor = preprocess_image(image_bytes)
        
        # Get prediction
        result = predict_ensemble(image_tensor)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'models_loaded': len(MODELS),
        'device': str(DEVICE)
    })

@app.route('/api/gpu-stats')
def gpu_stats():
    """GPU statistics endpoint"""
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        gpu_utilization = (gpu_memory_allocated / gpu_memory_total) * 100 if gpu_memory_total > 0 else 0
        
        return jsonify({
            'available': True,
            'allocated_gb': round(gpu_memory_allocated, 2),
            'reserved_gb': round(gpu_memory_reserved, 2),
            'total_gb': round(gpu_memory_total, 2),
            'utilization_percent': round(gpu_utilization, 1)
        })
    else:
        return jsonify({
            'available': False,
            'utilization_percent': 0
        })

@app.route('/api/gradcam', methods=['POST'])
def gradcam():
    """Generate Grad-CAM visualization for uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        image_tensor = preprocess_image(image_bytes)
        
        # Get prediction first to know which class to visualize
        result = predict_ensemble(image_tensor)
        predicted_class = CLASS_NAMES.index(result['prediction'])
        
        # Generate Grad-CAM for all models
        model_list = list(MODELS.values())
        model_names = list(MODELS.keys())
        
        visualizations = generate_ensemble_gradcam(
            model_list, 
            model_names, 
            image_tensor, 
            predicted_class
        )
        
        # Create grid visualization
        grid_image = create_gradcam_grid(visualizations, image_tensor)
        
        # Convert to base64 for JSON response
        _, buffer = cv2.imencode('.png', cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
        grid_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Also encode individual visualizations
        individual_vis = {}
        for name, vis in visualizations.items():
            _, buffer = cv2.imencode('.png', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            individual_vis[name] = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'grid': grid_base64,
            'individual': individual_vis,
            'predicted_class': result['prediction'],
            'confidence': result['confidence']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 80)
    print("GALAXY CLASSIFIER INFERENCE SERVER")
    print("=" * 80)
    
    # Load models
    load_models()
    
    print("\n🚀 Starting server at http://localhost:8080")
    print("=" * 80 + "\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
