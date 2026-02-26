# Galaxy Classifier - Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     GALAXY MORPHOLOGY CLASSIFIER                         │
│                        Production-Ready System                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACES                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │   Web App    │    │   REST API   │    │   CLI Tool   │             │
│  │              │    │              │    │              │             │
│  │  Port 5001   │    │  Port 8000   │    │   Terminal   │             │
│  │              │    │              │    │              │             │
│  │  - Upload    │    │  - /predict  │    │  - predict.py│             │
│  │  - Preview   │    │  - /batch    │    │  - Rich UI   │             │
│  │  - Results   │    │  - /health   │    │  - Tables    │             │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘             │
│         │                   │                   │                      │
└─────────┼───────────────────┼───────────────────┼──────────────────────┘
          │                   │                   │
          └───────────────────┴───────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────────────┐
│                       INFERENCE ENGINE                                  │
├─────────────────────────────┼─────────────────────────────────────────┤
│                             │                                          │
│                    ┌────────▼────────┐                                 │
│                    │  Image Processor │                                │
│                    │                  │                                │
│                    │  - Resize 69x69  │                                │
│                    │  - Normalize     │                                │
│                    │  - To Tensor     │                                │
│                    └────────┬─────────┘                                │
│                             │                                          │
│                    ┌────────▼────────┐                                 │
│                    │ Ensemble Models  │                                │
│                    │                  │                                │
│  ┌─────────────────┼──────────────────┼─────────────────┐             │
│  │                 │                  │                 │             │
│  ▼                 ▼                  ▼                 ▼             │
│ ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│ │ ResNet50 │  │DenseNet  │  │Efficient │  │Efficient │              │
│ │          │  │  121     │  │Net-B0 V1 │  │Net-B0 V2 │              │
│ │ 70.50%   │  │ 66.03%   │  │ 60.05%   │  │ 60.05%   │              │
│ └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│      │             │             │             │                      │
│      └─────────────┴─────────────┴─────────────┘                      │
│                    │                                                   │
│           ┌────────▼────────┐                                         │
│           │  Ensemble Avg   │                                         │
│           │                 │                                         │
│           │  - Average probs│                                         │
│           │  - Top-3        │                                         │
│           │  - Confidence   │                                         │
│           └────────┬────────┘                                         │
│                    │                                                   │
└────────────────────┼─────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            RESPONSE                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  {                                                                       │
│    "prediction": "Barred Spiral Galaxies",                              │
│    "confidence": 85.42,                                                 │
│    "top3": [...],                                                       │
│    "individual_models": {...},                                          │
│    "all_probabilities": {...}                                           │
│  }                                                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT OPTIONS                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │    Docker    │    │    Cloud     │    │      K8s     │             │
│  │              │    │              │    │              │             │
│  │  - Compose   │    │  - AWS ECS   │    │  - Pods      │             │
│  │  - Swarm     │    │  - GCP Run   │    │  - Services  │             │
│  │  - Standalone│    │  - Azure ACI │    │  - Ingress   │             │
│  └──────────────┘    └──────────────┘    └──────────────┘             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                         TESTING PYRAMID                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         ┌─────────────┐                                 │
│                         │ Integration │                                 │
│                         │   Tests     │                                 │
│                         │             │                                 │
│                         │  - API      │                                 │
│                         │  - E2E      │                                 │
│                         └─────────────┘                                 │
│                      ┌───────────────────┐                              │
│                      │   Unit Tests      │                              │
│                      │                   │                              │
│                      │  - Data Loading   │                              │
│                      │  - Models         │                              │
│                      │  - Inference      │                              │
│                      └───────────────────┘                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                         PROJECT STRUCTURE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  galaxy-morphology-classifier/                                          │
│  ├── src/                                                               │
│  │   ├── inference_server.py      ← Flask Web App                      │
│  │   ├── api_server.py            ← FastAPI REST API                   │
│  │   ├── predict.py               ← CLI Tool                           │
│  │   ├── train_optimized.py       ← Training (Ensemble)                │
│  │   ├── evaluate.py              ← Evaluation                         │
│  │   └── static_inference/        ← Web UI Assets                      │
│  │       ├── inference.html                                            │
│  │       ├── inference_style.css                                       │
│  │       └── inference_app.js                                          │
│  ├── tests/                                                             │
│  │   ├── test_classifier.py       ← Unit Tests                         │
│  │   └── test_api.py              ← Integration Tests                  │
│  ├── models/                       ← Trained Models (.pth)             │
│  ├── Dockerfile                    ← Container Definition               │
│  ├── docker-compose.yml            ← Multi-service Setup                │
│  ├── pytest.ini                    ← Test Configuration                 │
│  ├── requirements.txt              ← Dependencies                       │
│  ├── README.md                     ← Main Documentation                 │
│  ├── QUICKSTART.md                 ← Quick Start Guide                  │
│  ├── DEPLOYMENT.md                 ← Deployment Guide                   │
│  └── IMPLEMENTATION_SUMMARY.md     ← This Implementation                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                         TECHNOLOGY STACK                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Backend:                                                               │
│    • Python 3.11                                                        │
│    • PyTorch 2.5.1                                                      │
│    • FastAPI 0.104+                                                     │
│    • Flask 2.3+                                                         │
│                                                                          │
│  Frontend:                                                              │
│    • HTML5                                                              │
│    • CSS3 (Glassmorphism)                                               │
│    • Vanilla JavaScript                                                 │
│                                                                          │
│  DevOps:                                                                │
│    • Docker                                                             │
│    • Docker Compose                                                     │
│    • Pytest                                                             │
│                                                                          │
│  Models:                                                                │
│    • ResNet50                                                           │
│    • DenseNet121                                                        │
│    • EfficientNet-B0                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                            FEATURES                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ✅ Web Inference App         - Beautiful UI for classification         │
│  ✅ REST API                   - Production-ready FastAPI               │
│  ✅ CLI Tool                   - Terminal-based predictions             │
│  ✅ Docker Deployment          - Containerized solution                 │
│  ✅ Unit Tests                 - Comprehensive coverage                 │
│  ✅ Integration Tests          - API endpoint testing                   │
│  ✅ Documentation              - 3 detailed guides                      │
│  ✅ Ensemble Predictions       - 4 models averaging                     │
│  ✅ Batch Processing           - Multiple images at once                │
│  ✅ Health Checks              - Monitoring endpoints                   │
│  ✅ Auto Documentation         - Swagger/OpenAPI                        │
│  ✅ Error Handling             - Comprehensive validation               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                         PERFORMANCE METRICS                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Model Accuracy:        76.14% (Ensemble + TTA)                         │
│  Inference Time:        ~100-200ms (GPU)                                │
│  API Response Time:     ~150-300ms (including I/O)                      │
│  Docker Image Size:     ~2-3 GB (with models)                           │
│  Test Coverage:         30+ test cases                                  │
│  Lines of Code:         ~3,850+ (new features)                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Commands

```bash
# Web App
python src/inference_server.py

# REST API
python src/api_server.py

# CLI
python src/predict.py galaxy.jpg

# Docker
docker-compose up -d

# Tests
pytest --cov=src
```

## Galaxy Classes (10 Types)

1. Disturbed Galaxies
2. Merging Galaxies
3. Round Smooth Galaxies
4. In-between Round Smooth Galaxies
5. Cigar Shaped Smooth Galaxies
6. Barred Spiral Galaxies
7. Unbarred Tight Spiral Galaxies
8. Unbarred Loose Spiral Galaxies
9. Edge-on Galaxies without Bulge
10. Edge-on Galaxies with Bulge

---

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**License**: MIT
