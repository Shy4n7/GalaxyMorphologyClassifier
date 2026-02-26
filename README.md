# Galaxy Morphology Classifier 🌌

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![Vite](https://img.shields.io/badge/Frontend-Vite%2BReact-646CFF.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/Shy4n7/GalaxyMorphologyClassifier?style=social)](https://github.com/Shy4n7/GalaxyMorphologyClassifier)

An advanced, production-grade deep learning ecosystem for classifying galaxy morphology. This project implements a high-performance **weighted ensemble** of Convolutional Neural Networks (CNNs) nicknamed the **"Guardians of the Galaxy"**, achieving an elite **95.4% accuracy** on morphological classification.

> **Portfolio Project**: Showcases expertise in Deep Learning, Ensemble Methods, Explainable AI (XAI), and Modern Full-Stack ML Deployment with a "Guardians" themed dashboard.

---

## 🚀 Key Performance Metrics

| Metric | Performance | Status |
|--------|-------------|--------|
| **Ensemble Accuracy** | **95.4%** 🚀 | ✅ Target Met |
| **Baseline Accuracy** | 14.9% | ⏫ 6.4x Gain |
| **Best Single Model** | 94.2% (Gamora) | ⭐ SOTA |
| **Inference Latency** | ~150-200ms | ⚡ Real-time |
| **Ensemble Strategy** | Weighted Soft Voting | 🧠 Advanced |

---

## 🛡️ The "Guardians" Ensemble

The heart of the system is the **Ensemble Classifier**, composed of five specialized models trained for high precision:

1.  **🔴 STAR-LORD (ResNet-50)**: The "Charismatic Leader" architecture. CAPTURES deep residual features with **93.8%** accuracy.
2.  **🟢 GAMORA (DenseNet-121)**: The "Deadly Accurate" model. Uses dense connections for feature reuse, leading the group with **94.2%** accuracy.
3.  **🔘 DRAX (EfficientNet-V1)**: The "Literal Power" house. Compound-scaled for high efficiency, achieving **92.5%** accuracy.
4.  **🟠 ROCKET (EfficientNet-V2)**: The "Small Aggressor". Optimized version with faster training and inference, hitting **93.1%** accuracy.
5.  **🟤 GROOT (MobileNet-V3)**: The "I am MobileNet" lightweight champion. Fast, efficient, and robust at **92.5%** accuracy.

---

## 🛠️ Technology Stack

### Backend & ML
*   **Core**: Python 3.11, PyTorch 2.5.1
*   **Architectures**: ResNet50, EfficientNet (B0/B2), DenseNet121, MobileNetV3
*   **APIs**: FastAPI (Production API), Flask (Inference Interface)
*   **Explainability**: Grad-CAM for saliency mapping

### Frontend & Monitoring
*   **Framework**: React 19, TypeScript, Vite
*   **Visualizations**: Recharts (Dynamic training curves & model metrics)
*   **Styling**: Glassmorphic UI with "Galaxy" theme & particles.js animations

---

## 🧠 Core Features

### 1. Weighted Soft-Voting Ensemble
Each model's prediction is weighted for a final consensus, ensuring that the "Guardians" vote as a team for maximum classification reliability.

### 2. Explainable AI (Grad-CAM)
Transparency in astronomical classification. Real-time generation of heatmaps highlighting specific regions (spiral arms, bulges) that influence the model's decision.

### 3. Real-Time Ensemble Dashboard
A modern Vite-powered dashboard providing:
*   **Live Metrics**: CPU/GPU utilization and VRAM monitoring.
*   **Model Feed**: Real-time predictions from the active ensemble.
*   **Dynamic Visuals**: interactive graphs showing model weights and individual contributions.

---

## 📁 Project Structure

```bash
GalaxyClassifier/
├── src/
│   ├── inference_server.py    # Flask Web App for Inference & Grad-CAM
│   ├── api_server.py          # FastAPI Production REST API
│   ├── train_optimized.py     # Weighted ensemble training logic
│   ├── train_kfold.py         # K-Fold CV implementation
│   └── dashboard_server.py    # Monitoring telemetry backend
├── frontend/                  # React + Vite + TypeScript Dashboard
├── models/                    # Trained weights (.pth) - [Git Ignored]
├── tests/                     # Comprehensive Pytest suite
└── DEPLOYMENT.md              # Cloud (AWS/GCP/Azure) deployment guide
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/Shy4n7/GalaxyMorphologyClassifier.git
cd GalaxyMorphologyClassifier

# Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. Run the Ecosystem
```bash
# Start the Inference Server (Web UI + Grad-CAM)
python src/inference_server.py

# Start the Monitoring Dashboard
npm run dev --prefix frontend
```

---

## 📧 Contact

**Shyan** - [@Shy4n7](https://github.com/Shy4n7)  
**Email**: [shyanpaul7@gmail.com](mailto:shyanpaul7@gmail.com)  

---
*Built with ❤️ for Astronomy and Deep Learning.*
