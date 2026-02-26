"""
Galaxy Classifier REST API
FastAPI-based production-ready API for galaxy classification
"""

import os
import sys
import io
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import torchvision.transforms as transforms
import torchvision.models as models
import uvicorn

# Add src to path
sys.path.append(os.path.dirname(__file__))

# Initialize FastAPI app
app = FastAPI(
    title="Galaxy Morphology Classifier API",
    description="Deep learning API for classifying galaxy morphology using ensemble models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Response models
class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    top3: List[Dict[str, float]]
    individual_models: Dict[str, Dict[str, float]]
    all_probabilities: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    device: str
    version: str

class ModelInfo(BaseModel):
    name: str
    architecture: str
    parameters: str
    accuracy: str

# Model architectures (same as training)
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
    """Load all trained ensemble models"""
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
            try:
                model = OverfitModel(model_type=model_type, num_classes=10)
                state_dict = torch.load(path, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                model.eval()
                
                # Calculate parameters
                params = sum(p.numel() for p in model.parameters())
                param_str = f"{params/1e6:.1f}M"
                
                MODELS[name] = {
                    'model': model,
                    'architecture': model_type,
                    'parameters': param_str,
                    'accuracy': f"{acc:.2f}%"
                }
                loaded_count += 1
                print(f"✓ Loaded {name} (Acc: {acc:.2f}%)")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
        else:
            print(f"✗ No overfit model found for {model_type}")

    if not MODELS:
        print("⚠️ No models loaded! Please train models first.")
    else:
        print(f"✓ Successfully loaded {len(MODELS)} models")

def preprocess_image(image_bytes: bytes):
    """Preprocess uploaded image for model inference"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((69, 69)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor.to(DEVICE)

def predict_ensemble(image_tensor):
    """Run ensemble prediction with all models"""
    if not MODELS:
        raise RuntimeError("No models loaded")
    
    predictions = []
    model_results = {}
    
    with torch.no_grad():
        for name, model_info in MODELS.items():
            model = model_info['model']
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            predictions.append(probs)
            
            pred_class = np.argmax(probs)
            model_results[name] = {
                'class': CLASS_NAMES[pred_class],
                'confidence': float(probs[pred_class] * 100)
            }
    
    # Ensemble averaging
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

# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Galaxy Morphology Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /predict": "Classify galaxy image",
            "GET /models": "List available models",
            "GET /classes": "List galaxy classes"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if MODELS else "no_models",
        "models_loaded": len(MODELS),
        "device": str(DEVICE),
        "version": "1.0.0"
    }

@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """List all loaded models with their information"""
    return [
        {
            "name": name,
            "architecture": info['architecture'],
            "parameters": info['parameters'],
            "accuracy": info['accuracy']
        }
        for name, info in MODELS.items()
    ]

@app.get("/classes", tags=["Classes"])
async def list_classes():
    """List all galaxy classes"""
    return {
        "classes": CLASS_NAMES,
        "count": len(CLASS_NAMES)
    }

@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Classify a galaxy image
    
    - **file**: Image file (JPG, PNG, etc.)
    
    Returns:
    - **prediction**: Predicted galaxy class
    - **confidence**: Confidence percentage
    - **top3**: Top 3 predictions with confidences
    - **individual_models**: Predictions from each model
    - **all_probabilities**: Probabilities for all classes
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        
        # Get prediction
        result = predict_ensemble(image_tensor)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Classify multiple galaxy images in batch
    
    - **files**: List of image files
    
    Returns list of predictions
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    for file in files:
        try:
            image_bytes = await file.read()
            image_tensor = preprocess_image(image_bytes)
            result = predict_ensemble(image_tensor)
            results.append({
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return results

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("=" * 80)
    print("GALAXY CLASSIFIER REST API")
    print("=" * 80)
    load_models()
    print("=" * 80)

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
