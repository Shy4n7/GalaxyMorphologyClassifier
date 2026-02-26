# Galaxy Classifier - Deployment Guide 🚀

This guide covers deploying the Galaxy Morphology Classifier using Docker.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Running Services](#running-services)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required
- Docker (20.10+)
- Docker Compose (2.0+)
- Trained models in `models/` directory

### Optional
- NVIDIA Docker (for GPU support)
- 4GB+ RAM
- 10GB+ disk space

## Quick Start

### 1. Build Docker Image

```bash
# Build the image
docker build -t galaxy-classifier:latest .

# Or use docker-compose
docker-compose build
```

### 2. Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Access Services

- **REST API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Web App**: http://localhost:5001

## Docker Deployment

### Single Container (API Only)

```bash
# Run API server
docker run -d \
  --name galaxy-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  galaxy-classifier:latest \
  python src/api_server.py
```

### Single Container (Web App Only)

```bash
# Run web inference server
docker run -d \
  --name galaxy-webapp \
  -p 5001:5001 \
  -v $(pwd)/models:/app/models:ro \
  galaxy-classifier:latest \
  python src/inference_server.py
```

### With GPU Support

```bash
# Install NVIDIA Docker runtime first
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Run with GPU
docker run -d \
  --gpus all \
  --name galaxy-api-gpu \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  galaxy-classifier:latest \
  python src/api_server.py
```

## Running Services

### REST API Server

The FastAPI server provides a production-ready REST API.

**Start:**
```bash
# Local (without Docker)
python src/api_server.py

# Docker
docker-compose up -d api
```

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /models` - List loaded models
- `GET /classes` - List galaxy classes
- `POST /predict` - Classify single image
- `POST /predict/batch` - Classify multiple images

**Example Request:**
```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@galaxy.jpg"

# Using Python
import requests

with open('galaxy.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    print(response.json())
```

### Web Inference App

The Flask-based web interface for interactive classification.

**Start:**
```bash
# Local (without Docker)
python src/inference_server.py

# Docker
docker-compose up -d webapp
```

**Access:** http://localhost:5001

### CLI Tool

Command-line interface for quick predictions.

**Usage:**
```bash
# Classify single image
python src/predict.py galaxy.jpg

# Force CPU inference
python src/predict.py galaxy.jpg --cpu

# Get help
python src/predict.py --help
```

## API Documentation

### Interactive API Docs

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Response Format

```json
{
  "prediction": "Barred Spiral Galaxies",
  "confidence": 85.42,
  "top3": [
    {
      "class": "Barred Spiral Galaxies",
      "confidence": 85.42
    },
    {
      "class": "Unbarred Tight Spiral Galaxies",
      "confidence": 8.31
    },
    {
      "class": "Unbarred Loose Spiral Galaxies",
      "confidence": 3.27
    }
  ],
  "individual_models": {
    "resnet50": {
      "class": "Barred Spiral Galaxies",
      "confidence": 88.5
    },
    "densenet121": {
      "class": "Barred Spiral Galaxies",
      "confidence": 82.3
    }
  },
  "all_probabilities": {
    "Disturbed Galaxies": 0.12,
    "Merging Galaxies": 0.45,
    ...
  }
}
```

## Docker Compose Configuration

### Services

The `docker-compose.yml` defines two services:

1. **api** - FastAPI REST API (port 8000)
2. **webapp** - Flask web interface (port 5001)

### Resource Limits

Default limits per service:
- CPU: 2 cores (limit), 1 core (reservation)
- Memory: 4GB (limit), 2GB (reservation)

Adjust in `docker-compose.yml` if needed.

### Volumes

- `./models:/app/models:ro` - Read-only model files
- `./data:/app/data:ro` - Read-only dataset
- `./logs:/app/logs` - Persistent logs

## Environment Variables

### Available Variables

```bash
# Python
PYTHONUNBUFFERED=1          # Disable output buffering

# Custom (add to docker-compose.yml)
MODEL_PATH=/app/models       # Model directory
LOG_LEVEL=info              # Logging level
```

## Production Deployment

### Cloud Platforms

#### AWS (EC2 + ECR)

```bash
# 1. Build and tag
docker build -t galaxy-classifier:latest .

# 2. Tag for ECR
docker tag galaxy-classifier:latest \
  <account-id>.dkr.ecr.<region>.amazonaws.com/galaxy-classifier:latest

# 3. Push to ECR
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/galaxy-classifier:latest

# 4. Deploy on EC2
# SSH into EC2 instance and pull image
docker pull <account-id>.dkr.ecr.<region>.amazonaws.com/galaxy-classifier:latest
docker run -d -p 8000:8000 ...
```

#### Google Cloud Run

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/<project-id>/galaxy-classifier

# 2. Deploy to Cloud Run
gcloud run deploy galaxy-classifier \
  --image gcr.io/<project-id>/galaxy-classifier \
  --platform managed \
  --port 8000 \
  --memory 4Gi
```

#### Azure Container Instances

```bash
# 1. Build and push to ACR
az acr build --registry <registry-name> \
  --image galaxy-classifier:latest .

# 2. Deploy to ACI
az container create \
  --resource-group <rg-name> \
  --name galaxy-classifier \
  --image <registry-name>.azurecr.io/galaxy-classifier:latest \
  --dns-name-label galaxy-classifier \
  --ports 8000
```

## Troubleshooting

### Common Issues

#### 1. Models Not Loading

**Problem:** "No models loaded" error

**Solution:**
```bash
# Ensure models exist
ls -lh models/

# Check volume mount
docker run --rm -v $(pwd)/models:/app/models galaxy-classifier:latest ls -lh /app/models
```

#### 2. Out of Memory

**Problem:** Container crashes with OOM

**Solution:**
```bash
# Increase memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G  # Increase from 4G
```

#### 3. Port Already in Use

**Problem:** "Address already in use" error

**Solution:**
```bash
# Find process using port
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill process or change port
docker-compose up -d --force-recreate
```

#### 4. Slow Inference

**Problem:** Predictions take too long

**Solution:**
- Use GPU: Add `--gpus all` flag
- Reduce batch size
- Use single model instead of ensemble

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check container health
docker ps
docker inspect galaxy-api | grep Health

# View logs
docker logs galaxy-api -f
```

### Performance Monitoring

```bash
# Monitor resource usage
docker stats

# Container metrics
docker inspect galaxy-api --format='{{.State.Health.Status}}'
```

## Scaling

### Horizontal Scaling

```bash
# Scale API service
docker-compose up -d --scale api=3

# Use load balancer (nginx, traefik, etc.)
```

### Kubernetes Deployment

See `k8s/` directory for Kubernetes manifests (if available).

## Security

### Best Practices

1. **Use non-root user** (add to Dockerfile)
2. **Scan images** for vulnerabilities
3. **Use secrets** for sensitive data
4. **Enable HTTPS** in production
5. **Implement rate limiting**

### Example: Add Non-Root User

```dockerfile
# Add to Dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

## Maintenance

### Updating Models

```bash
# 1. Stop services
docker-compose down

# 2. Update models in models/
cp new_model.pth models/

# 3. Restart services
docker-compose up -d
```

### Backup

```bash
# Backup models
tar -czf models-backup.tar.gz models/

# Backup logs
tar -czf logs-backup.tar.gz logs/
```

## Support

For issues or questions:
- GitHub Issues: [Your Repo URL]
- Email: [Your Email]

---

**Built with ❤️ for astronomy and deep learning**
