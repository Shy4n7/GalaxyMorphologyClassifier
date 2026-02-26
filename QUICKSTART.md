# Galaxy Classifier - Quick Start Guide 🚀

Get started with the Galaxy Morphology Classifier in minutes!

## Table of Contents
1. [Installation](#installation)
2. [Using Pre-trained Models](#using-pre-trained-models)
3. [Inference Options](#inference-options)
4. [Training Your Own Models](#training-your-own-models)
5. [Deployment](#deployment)

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/Shy4n7/Galaxy_Morphology_Classifier.git
cd Galaxy_Morphology_Classifier
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 3: Download Models

Pre-trained models are available in the `models/` directory or from GitHub Releases.

---

## Using Pre-trained Models

### Option 1: Web Interface (Easiest) 🌐

**Start the web app:**
```bash
python src/inference_server.py
```

**Then:**
1. Open http://localhost:5001 in your browser
2. Drag and drop a galaxy image
3. Click "Classify Galaxy"
4. View results with confidence scores!

**Features:**
- Beautiful, modern UI
- Drag-and-drop upload
- Real-time predictions
- Detailed model breakdowns
- Probability distributions

---

### Option 2: CLI Tool (Fast) ⚡

**Classify an image:**
```bash
python src/predict.py path/to/galaxy.jpg
```

**Example output:**
```
┌─────────────────────────────────────────┐
│ 🌌 Galaxy Classification Result        │
│                                         │
│ Barred Spiral Galaxies                  │
│ Confidence: 85.42%                      │
└─────────────────────────────────────────┘

Top 3 Predictions:
┌──────┬────────────────────────────────┬────────────┐
│ Rank │ Galaxy Type                    │ Confidence │
├──────┼────────────────────────────────┼────────────┤
│ #1   │ Barred Spiral Galaxies         │ 85.42%     │
│ #2   │ Unbarred Tight Spiral Galaxies │ 8.31%      │
│ #3   │ Unbarred Loose Spiral Galaxies │ 3.27%      │
└──────┴────────────────────────────────┴────────────┘
```

**Options:**
```bash
# Force CPU inference (if no GPU)
python src/predict.py galaxy.jpg --cpu

# Get help
python src/predict.py --help
```

---

### Option 3: REST API (Production) 🔌

**Start the API server:**
```bash
python src/api_server.py
```

**Interactive API docs:** http://localhost:8000/docs

**Python example:**
```python
import requests

# Classify an image
with open('galaxy.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Top 3: {result['top3']}")
```

**cURL example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@galaxy.jpg"
```

**JavaScript example:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Prediction:', data.prediction);
    console.log('Confidence:', data.confidence);
});
```

---

## Inference Options

### All Available Methods

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Web App** | Interactive use, demos | Beautiful UI, easy to use | Requires browser |
| **CLI Tool** | Batch processing, scripts | Fast, scriptable | Terminal only |
| **REST API** | Production, integration | Scalable, language-agnostic | Requires setup |
| **Docker** | Deployment, consistency | Portable, isolated | Requires Docker |

### Galaxy Classes

The classifier can identify 10 types of galaxies:

1. **Disturbed Galaxies** - Irregular, chaotic structure
2. **Merging Galaxies** - Two or more galaxies colliding
3. **Round Smooth Galaxies** - Elliptical, no spiral arms
4. **In-between Round Smooth Galaxies** - Transitional morphology
5. **Cigar Shaped Smooth Galaxies** - Elongated elliptical
6. **Barred Spiral Galaxies** - Spiral with central bar
7. **Unbarred Tight Spiral Galaxies** - Tightly wound spiral arms
8. **Unbarred Loose Spiral Galaxies** - Loosely wound spiral arms
9. **Edge-on Galaxies without Bulge** - Disk viewed edge-on
10. **Edge-on Galaxies with Bulge** - Disk with central bulge

---

## Training Your Own Models

### Step 1: Download Dataset

```bash
# Download Galaxy10 DECals dataset (2.54 GB)
# From: http://astro.utoronto.ca/~bovy/Galaxy10/Galaxy10_DECals.h5

# Place in data/ directory
mkdir -p data
# Move Galaxy10_DECals.h5 to data/
```

### Step 2: Train Models

**Single model (fastest):**
```bash
python src/train_pytorch.py
```

**Ensemble (3 models):**
```bash
python src/train_ensemble.py
```

**Optimized ensemble (5 models + TTA) - Best results:**
```bash
python src/train_optimized.py
```

### Step 3: Evaluate

```bash
python src/evaluate.py
```

**Training time:**
- Single model: ~30-45 min (GPU)
- Ensemble: ~2-3 hours (GPU)
- CPU training: 10-20x slower

---

## Deployment

### Docker (Recommended for Production)

**Quick start:**
```bash
# Build image
docker build -t galaxy-classifier:latest .

# Run API server
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  galaxy-classifier:latest

# Or use docker-compose (runs both API and web app)
docker-compose up -d
```

**Access services:**
- API: http://localhost:8000
- Web App: http://localhost:5001
- API Docs: http://localhost:8000/docs

### Cloud Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guides on:
- AWS (EC2, ECS, Lambda)
- Google Cloud (Cloud Run, GKE)
- Azure (Container Instances, AKS)
- Heroku, DigitalOcean, etc.

---

## Testing

**Run tests:**
```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific tests
pytest tests/test_classifier.py -v
```

---

## Troubleshooting

### Common Issues

#### 1. "No models loaded"

**Problem:** Models not found in `models/` directory

**Solution:**
```bash
# Check if models exist
ls -lh models/

# Download from GitHub Releases or train models
python src/train_optimized.py
```

#### 2. CUDA out of memory

**Problem:** GPU memory insufficient

**Solution:**
```bash
# Use CPU instead
python src/predict.py galaxy.jpg --cpu

# Or reduce batch size in training
```

#### 3. Module not found

**Problem:** Missing dependencies

**Solution:**
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

#### 4. Port already in use

**Problem:** Port 8000 or 5001 already occupied

**Solution:**
```bash
# Find and kill process
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Or change port in code
```

---

## Performance Tips

### For Faster Inference:

1. **Use GPU:** 10-50x faster than CPU
   ```bash
   # Check GPU availability
   python src/verify_gpu.py
   ```

2. **Use Single Model:** Instead of ensemble
   ```python
   # Load only ResNet50 (fastest single model)
   ```

3. **Batch Processing:** Process multiple images at once
   ```bash
   # Use API batch endpoint
   POST /predict/batch
   ```

### For Better Accuracy:

1. **Use Full Ensemble:** All 5 models + TTA
2. **Preprocess Images:** Ensure good quality
3. **Use Original Resolution:** Avoid heavy compression

---

## Next Steps

- ⭐ **Star the repo** if you find it useful!
- 📖 Read the full [README.md](README.md) for detailed documentation
- 🚀 Check [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- 🐛 Report issues on GitHub
- 🤝 Contribute improvements via Pull Requests

---

## Quick Reference

### Commands Cheat Sheet

```bash
# Inference
python src/inference_server.py          # Web app
python src/api_server.py                # REST API
python src/predict.py galaxy.jpg        # CLI

# Training
python src/train_optimized.py           # Best results
python src/train_ensemble.py            # 3-model ensemble
python src/train_pytorch.py             # Single model

# Evaluation
python src/evaluate.py                  # Evaluate model

# Testing
pytest                                  # Run tests
pytest --cov=src                        # With coverage

# Docker
docker-compose up -d                    # Start all services
docker-compose down                     # Stop services
docker-compose logs -f                  # View logs

# Utilities
python src/verify_gpu.py                # Check GPU
python src/diagnose_data.py             # Check dataset
```

---

**Happy classifying! 🌌**

*For questions or issues, please open a GitHub issue.*
