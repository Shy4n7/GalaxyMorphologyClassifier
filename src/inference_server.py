"""
Galaxy Classifier Inference Server
3-model ensemble: ConvNeXt-Tiny + ResNeXt-50 + DenseNet161 (v3 weights)
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

sys.path.append(os.path.dirname(__file__))
from gradcam_utils import generate_gradcam, generate_ensemble_gradcam, create_gradcam_grid

app = Flask(__name__, static_folder='static')
CORS(app)

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
    "Edge-on Galaxies with Bulge",
]

NUM_CLASSES = 10
IMG_SIZE    = 224
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Weights proportional to TTA test accuracy: ConvNeXt 87.97, ResNeXt 86.40, DenseNet 86.28
ENSEMBLE_WEIGHTS = [87.97, 86.40, 86.28]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# TTA transforms matching train_optimized_v3.py
TTA_TRANSFORMS = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[3]),
    lambda x: torch.flip(x, dims=[2]),
    lambda x: torch.rot90(x, 1, dims=[2, 3]),
    lambda x: torch.rot90(x, 2, dims=[2, 3]),
    lambda x: torch.rot90(x, 3, dims=[2, 3]),
    lambda x: torch.flip(torch.rot90(x, 1, dims=[2, 3]), dims=[3]),
    lambda x: torch.flip(torch.rot90(x, 3, dims=[2, 3]), dims=[3]),
]

MODELS: dict = {}


# ---------------------------------------------------------------------------
# Model architectures (must match train_optimized_v3.py exactly)
# ---------------------------------------------------------------------------

def build_convnext_tiny() -> nn.Module:
    m = models.convnext_tiny(weights=None)
    in_f = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(512, NUM_CLASSES),
    )
    return m


def build_resnext50() -> nn.Module:
    m = models.resnext50_32x4d(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, NUM_CLASSES),
    )
    return m


def build_densenet161() -> nn.Module:
    m = models.densenet161(weights=None)
    in_f = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, NUM_CLASSES),
    )
    return m


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _latest(prefix: str) -> str | None:
    """Return the most recently modified .pth matching models/best_v3_{prefix}_*.pth"""
    d = 'models'
    candidates = [
        f for f in os.listdir(d)
        if f.startswith(f'best_v3_{prefix}_') and f.endswith('.pth')
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    return os.path.join(d, candidates[0])


def load_models():
    global MODELS

    print(f"Device: {DEVICE}")

    configs = [
        ('convnext',   build_convnext_tiny,  'convnext'),
        ('resnext50',  build_resnext50,       'resnext50'),
        ('densenet161', build_densenet161,    'densenet161'),
    ]

    for name, builder, prefix in configs:
        path = _latest(prefix)
        if path is None:
            print(f"  [SKIP] No weights found for {name}")
            continue
        try:
            m = builder()
            ckpt = torch.load(path, map_location=DEVICE)
            state = ckpt.get('model', ckpt)
            m.load_state_dict(state)
            m.to(DEVICE)
            m.eval()
            MODELS[name] = m
            print(f"  [OK] {name} <- {os.path.basename(path)}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    print(f"Loaded {len(MODELS)}/3 models")


# ---------------------------------------------------------------------------
# Preprocessing + inference
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    base_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    x = base_transform(img).unsqueeze(0).to(DEVICE)
    return x


def predict_ensemble(x: torch.Tensor) -> dict:
    if not MODELS:
        raise RuntimeError("No models loaded")

    # Build 8-view TTA batch once
    tta_batch = torch.cat([t(x) for t in TTA_TRANSFORMS], dim=0)  # (8, 3, H, W)

    all_probs  = []
    model_results = {}
    weights    = list(ENSEMBLE_WEIGHTS)

    with torch.no_grad():
        for (name, model), w in zip(MODELS.items(), weights):
            out   = model(tta_batch)                              # (8, 10)
            probs = torch.softmax(out, dim=1).mean(dim=0)        # (10,) TTA average
            probs_np = probs.cpu().numpy()
            all_probs.append((probs_np, w))

            pred = int(np.argmax(probs_np))
            model_results[name] = {
                'class':      CLASS_NAMES[pred],
                'confidence': float(probs_np[pred] * 100),
            }

    # Weighted soft voting
    total_w = sum(w for _, w in all_probs)
    ensemble = sum(p * (w / total_w) for p, w in all_probs)

    pred_class = int(np.argmax(ensemble))
    top3 = [
        {'class': CLASS_NAMES[i], 'confidence': float(ensemble[i] * 100)}
        for i in np.argsort(ensemble)[-3:][::-1]
    ]

    return {
        'prediction':        CLASS_NAMES[pred_class],
        'confidence':        float(ensemble[pred_class] * 100),
        'top3':              top3,
        'individual_models': model_results,
        'all_probabilities': {CLASS_NAMES[i]: float(ensemble[i] * 100) for i in range(NUM_CLASSES)},
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        f = request.files['image']
        if f.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        result = predict_ensemble(preprocess_image(f.read()))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    return jsonify({
        'status':        'ok',
        'models_loaded': len(MODELS),
        'model_names':   list(MODELS.keys()),
        'device':        str(DEVICE),
    })


@app.route('/api/gpu-stats')
def gpu_stats():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1024**3
        resv  = torch.cuda.memory_reserved(0)  / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return jsonify({
            'available':          True,
            'allocated_gb':       round(alloc, 2),
            'reserved_gb':        round(resv,  2),
            'total_gb':           round(total, 2),
            'utilization_percent': round(alloc / total * 100 if total else 0, 1),
        })
    return jsonify({'available': False, 'utilization_percent': 0})


@app.route('/api/gradcam', methods=['POST'])
def gradcam():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        f = request.files['image']
        if f.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        image_bytes  = f.read()
        image_tensor = preprocess_image(image_bytes)
        result       = predict_ensemble(image_tensor)
        pred_class   = CLASS_NAMES.index(result['prediction'])

        model_list  = list(MODELS.values())
        model_names = list(MODELS.keys())

        visualizations = generate_ensemble_gradcam(
            model_list, model_names, image_tensor, pred_class
        )
        grid_image = create_gradcam_grid(visualizations, image_tensor)

        _, buf = cv2.imencode('.png', cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
        grid_b64 = base64.b64encode(buf).decode('utf-8')

        individual_vis = {}
        for name, vis in visualizations.items():
            _, buf = cv2.imencode('.png', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            individual_vis[name] = base64.b64encode(buf).decode('utf-8')

        return jsonify({
            'grid':            grid_b64,
            'individual':      individual_vis,
            'predicted_class': result['prediction'],
            'confidence':      result['confidence'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("GALAXY CLASSIFIER  |  3-model ensemble (v3)")
    print("=" * 60)
    load_models()
    print(f"\nStarting server at http://localhost:8080")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=8080, debug=False)
