"""
Galaxy Classifier API
3-model ensemble: ConvNeXt-Tiny + ResNeXt-50 + DenseNet161 (v3 weights)
"""

import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Galaxy Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

NUM_CLASSES      = 10
IMG_SIZE         = 224
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENSEMBLE_WEIGHTS = [87.97, 86.40, 86.28]
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]

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
# Architectures (must match train_optimized_v3.py exactly)
# ---------------------------------------------------------------------------

def _build_convnext() -> nn.Module:
    m = models.convnext_tiny(weights=None)
    in_f = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 512), nn.GELU(),
        nn.Dropout(0.2), nn.Linear(512, NUM_CLASSES),
    )
    return m


def _build_resnext50() -> nn.Module:
    m = models.resnext50_32x4d(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 512), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(512, NUM_CLASSES),
    )
    return m


def _build_densenet161() -> nn.Module:
    m = models.densenet161(weights=None)
    in_f = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 512), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(512, NUM_CLASSES),
    )
    return m


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _latest_checkpoint(prefix: str) -> str | None:
    d = "models"
    candidates = [
        f for f in os.listdir(d)
        if f.startswith(f"best_v3_{prefix}_") and f.endswith(".pth")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    return os.path.join(d, candidates[0])


def load_models() -> None:
    global MODELS
    print(f"Device: {DEVICE}")
    configs = [
        ("convnext",    _build_convnext,    "convnext"),
        ("resnext50",   _build_resnext50,   "resnext50"),
        ("densenet161", _build_densenet161, "densenet161"),
    ]
    for name, builder, prefix in configs:
        path = _latest_checkpoint(prefix)
        if not path:
            print(f"  [SKIP] No weights for {name}")
            continue
        try:
            m = builder()
            ckpt = torch.load(path, map_location=DEVICE)
            m.load_state_dict(ckpt.get("model", ckpt))
            m.to(DEVICE).eval()
            MODELS[name] = m
            print(f"  [OK] {name} <- {os.path.basename(path)}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
    print(f"Loaded {len(MODELS)}/3 models")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

_base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _base_transform(img).unsqueeze(0).to(DEVICE)


def _predict(x: torch.Tensor) -> dict:
    if not MODELS:
        raise RuntimeError("No models loaded")

    tta_batch = torch.cat([t(x) for t in TTA_TRANSFORMS], dim=0)

    all_probs: list[tuple[np.ndarray, float]] = []
    model_results: dict = {}

    with torch.no_grad():
        for (name, model), w in zip(MODELS.items(), ENSEMBLE_WEIGHTS):
            probs = torch.softmax(model(tta_batch), dim=1).mean(dim=0).cpu().numpy()
            all_probs.append((probs, w))
            pred = int(np.argmax(probs))
            model_results[name] = {
                "class": CLASS_NAMES[pred],
                "confidence": float(probs[pred] * 100),
            }

    total_w  = sum(w for _, w in all_probs)
    ensemble = sum(p * (w / total_w) for p, w in all_probs)

    pred_class = int(np.argmax(ensemble))
    top3 = [
        {"class": CLASS_NAMES[i], "confidence": float(ensemble[i] * 100)}
        for i in np.argsort(ensemble)[-3:][::-1]
    ]

    return {
        "prediction":        CLASS_NAMES[pred_class],
        "confidence":        float(ensemble[pred_class] * 100),
        "top3":              top3,
        "individual_models": model_results,
        "all_probabilities": {CLASS_NAMES[i]: float(ensemble[i] * 100) for i in range(NUM_CLASSES)},
    }


def _gradcam_heatmap(model: nn.Module, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Simple gradient-based CAM without external lib dependency."""
    activations: list[torch.Tensor] = []
    gradients:   list[torch.Tensor] = []

    # Hook last conv-like layer
    def _fwd(_, __, out): activations.append(out)
    def _bwd(_, __, go):  gradients.append(go[0])

    # Find last Conv2d
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d found")

    fh = last_conv.register_forward_hook(_fwd)
    bh = last_conv.register_full_backward_hook(_bwd)

    out = model(x)
    model.zero_grad()
    out[0, class_idx].backward()

    fh.remove()
    bh.remove()

    act  = activations[0].squeeze(0).detach().cpu().numpy()  # (C, H, W)
    grad = gradients[0].squeeze(0).detach().cpu().numpy()    # (C, H, W)

    weights = grad.mean(axis=(1, 2))  # (C,)
    cam = np.maximum((weights[:, None, None] * act).sum(0), 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def _overlay(cam: np.ndarray, x: torch.Tensor) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std  = np.array(IMAGENET_STD)[:, None, None]
    raw  = x.squeeze(0).cpu().numpy() * std + mean
    raw  = np.clip(raw.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

    return cv2.addWeighted(raw, 0.5, heatmap, 0.5, 0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    load_models()


@app.get("/api/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": len(MODELS),
        "model_names":   list(MODELS.keys()),
        "device":        str(DEVICE),
    }


@app.get("/api/gpu-stats")
def gpu_stats():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return {
            "available":           True,
            "allocated_gb":        round(alloc, 2),
            "total_gb":            round(total, 2),
            "utilization_percent": round(alloc / total * 100 if total else 0, 1),
        }
    return {"available": False, "utilization_percent": 0}


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    try:
        result = _predict(_preprocess(await image.read()))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/gradcam")
async def gradcam(image: UploadFile = File(...)):
    try:
        image_bytes  = await image.read()
        x            = _preprocess(image_bytes)
        result       = _predict(x)
        pred_idx     = CLASS_NAMES.index(result["prediction"])

        panels: list[np.ndarray] = []
        labels: list[str] = []

        for name, model in MODELS.items():
            cam     = _gradcam_heatmap(model, x.clone(), pred_idx)
            overlay = _overlay(cam, x)
            panels.append(overlay)
            labels.append(name)

        # Stack panels horizontally
        grid = np.concatenate(panels, axis=1)

        _, buf    = cv2.imencode(".png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        grid_b64  = base64.b64encode(buf).decode()

        individual: dict[str, str] = {}
        for name, panel in zip(labels, panels):
            _, buf = cv2.imencode(".png", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            individual[name] = base64.b64encode(buf).decode()

        return {
            "grid":            grid_b64,
            "individual":      individual,
            "predicted_class": result["prediction"],
            "confidence":      result["confidence"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("GALAXY CLASSIFIER API  |  3-model ensemble (v3)")
    print("=" * 60)
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
