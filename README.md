# Galaxy Morphology Classifier 🌌

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/Shy4n7/Galaxy_Morphology_Classifier?style=social)](https://github.com/Shy4n7/Galaxy_Morphology_Classifier)

An advanced deep learning system for classifying galaxy morphology using the **Galaxy10 DECals** dataset. This project implements a state-of-the-art **weighted ensemble** of Convolutional Neural Networks (CNNs), achieving **76.14% test accuracy** through transfer learning, test-time augmentation (TTA), and Grad-CAM visualization.

> **Portfolio Project**: Showcases expertise in Deep Learning, Ensemble Methods, Explainable AI (XAI), and Production ML Deployment.

---

## 🎯 Key Achievements

| Metric | Performance |
|--------|-------------|
| **Final Test Accuracy** | **76.14%** 🚀 |
| **Baseline Accuracy** | 14.9% |
| **Improvement Factor** | **5.1x** over baseline |
| **Model Diversity** | 5-Model Weighted Ensemble |
| **Best Single Model** | ResNet50 (70.50%) |
| **Deployment** | Dockerized + REST API + Web UI |

---

## 🌟 Technical Highlights

### 🧠 Ensemble Architecture
The core system leverages a **weighted soft-voting ensemble** of five diverse architectures:
*   **ResNet50**: Captures deep residual features.
*   **EfficientNet-B0 & B2**: Compound-scaled models for high efficiency.
*   **DenseNet121**: Maximizes feature reuse via dense connections.
*   **MobileNetV3**: Lightweight architecture for faster inference path.

### 🔬 Advanced ML Techniques
*   **Transfer Learning**: Pre-trained ImageNet weights for robust feature extraction.
*   **Test-Time Augmentation (TTA)**: 4x rotational augmentation during inference to ensure rotation-invariant predictions.
*   **Weighted Loss & Class Balancing**: Handles the significant class imbalance in the Galaxy10 dataset.
*   **Explainable AI (Grad-CAM)**: Integrated Grad-CAM support to visualize which regions of the galaxy the models focus on for classification.
*   **K-Fold Cross-Validation**: Robust evaluation and model selection using 5-fold splits.

### 🌐 Production-Ready Ecosystem
*   **REST API (FastAPI)**: Low-latency inference endpoints with batch processing support.
*   **Inference Server (Flask)**: Interactive web application with drag-and-drop image classification.
*   **Dynamic Dashboard**: Real-time system monitoring (CPU/GPU) and model status tracking.
*   **Dockerized Deployment**: Fully containerized environment with `docker-compose` support.

---

## 📁 Project Structure

```bash
GalaxyClassifier/
├── src/
│   ├── inference_server.py    # Flask Web App for Inference & Grad-CAM
│   ├── api_server.py          # FastAPI Production REST API
│   ├── train_optimized.py     # Best performing ensemble training script
│   ├── train_kfold.py         # K-Fold CV training implementation
│   ├── gradcam_utils.py       # Grad-CAM visualization logic
│   └── dashboard_server.py    # System & Model monitoring dashboard
├── models/                    # Trained .pth model weights (Git Ignored)
├── data/                      # Dataset (Galaxy10_DECals.h5)
├── tests/                     # Comprehensive Pytest suite
├── notebooks/                 # EDA and experimentation (optional)
├── DEPLOYMENT.md              # Detailed guide for Cloud/On-prem deploy
└── ARCHITECTURE.md            # In-depth technical architecture
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Shy4n7/Galaxy_Morphology_Classifier.git
cd Galaxy_Morphology_Classifier

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
Download the [Galaxy10 DECals](http://astro.utoronto.ca/~bovy/Galaxy10/) dataset and place the `Galaxy10_DECals.h5` file in the `data/` directory.

### 3. Running the Inference App
Experience the classifier with a beautiful web interface:

```bash
python src/inference_server.py
```
Visit `http://localhost:8080` to upload galaxy images and see Grad-CAM visualizations!

---

## 📊 Results & Visualizations

### Confusion Matrix
The ensemble shows strong performance across all 10 classes, particularly on the most frequent "Round Smooth" and "Barred Spiral" types.

### Grad-CAM Heatmaps
The system provides transparency by highlighting the specific spiral arms, bulges, or disturbances that triggered a classification.

---

## 💻 Hardware Requirements

*   **Minimum**: 8GB RAM, Quad-core CPU (Inference only).
*   **Recommended**: NVIDIA GPU (4GB+ VRAM) with CUDA 12.1+ for training and fast inference.
*   **Tested on**: RTX 3050 (4GB Mobile), achieveing ~200ms inference time with full TTA.

---

## 🤝 Contributing & License

Contributions are welcome! Please feel free to submit a Pull Request.
Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 📧 Contact

**Shyan** - [@Shy4n7](https://github.com/Shy4n7)  
**Email**: shyanpaul7@gmail.com  
**Project Link**: [https://github.com/Shy4n7/Galaxy_Morphology_Classifier](https://github.com/Shy4n7/Galaxy_Morphology_Classifier)

---
*Built with ❤️ for Astronomy and Deep Learning.*
