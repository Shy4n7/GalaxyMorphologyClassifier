# Galaxy Morphology Classifier 🌌

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep learning ensemble classifier for galaxy morphology using the Galaxy10 DECals dataset. Achieves **76.14% test accuracy** using transfer learning with EfficientNet, ResNet, and DenseNet architectures.

> **Portfolio Project**: Demonstrates expertise in deep learning, ensemble methods, transfer learning, and production ML workflows.

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Final Accuracy** | **76.14%** |
| **Baseline** | 14.9% |
| **Improvement** | **5.1x** |
| **Models** | 5-model weighted ensemble |
| **Best Single Model** | ResNet50 (70.50%) |
| **Training Time** | ~2-3 hours on RTX 3050 |

## 📊 Performance Progression

```
Initial (TensorFlow CPU)     →  14.9%
Single Model (PyTorch GPU)   →  64.56%  (4.3x improvement)
Ensemble (3 models)          →  67.08%  (4.5x improvement)
Optimized Ensemble + TTA     →  76.14%  (5.1x improvement) ⭐
```

## 🌟 Technical Highlights

- **Transfer Learning**: Fine-tuned EfficientNet-B0/B2, ResNet50, DenseNet121
- **Ensemble Learning**: Weighted voting based on validation performance
- **Test-Time Augmentation**: 5x augmented predictions for robustness
- **Class Balancing**: Weighted loss for imbalanced dataset
- **GPU Acceleration**: Full PyTorch CUDA support with mixed precision (AMP)
- **Advanced Optimization**: OneCycleLR scheduler, label smoothing (0.1)
- **Production Ready**: Modular code, comprehensive documentation

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.11+
NVIDIA GPU with CUDA support (optional but recommended)
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/galaxy-morphology-classifier.git
cd galaxy-morphology-classifier

# Install dependencies
pip install -r requirements.txt

# Download Galaxy10 dataset
# Place Galaxy10_DECals.h5 in data/ directory
# Download from: http://astro.utoronto.ca/~bovy/Galaxy10/Galaxy10_DECals.h5
```

### Training

```bash
# Train single model (PyTorch GPU)
python src/train_pytorch.py

# Train ensemble (3 models)
python src/train_ensemble.py

# Train optimized ensemble (5 models + TTA) - Best Results
python src/train_optimized.py
```

### Using Pre-trained Models

Pre-trained models are available in **GitHub Releases**:

```python
import torch
from src.train_ensemble import ResNetModel

# Download models from GitHub Releases first
# Then load:
model = ResNetModel().cuda()
model.load_state_dict(torch.load('models/ensemble_resnet50.pth'))
model.eval()
```

## 📁 Project Structure

```
galaxy-morphology-classifier/
├── data/
│   ├── Galaxy10_DECals.h5          # Dataset (download separately)
│   └── README.md                   # Dataset information
├── src/
│   ├── load_data.py                # Data loading & preprocessing
│   ├── model.py                    # Model architectures (TensorFlow)
│   ├── train.py                    # TensorFlow training (baseline)
│   ├── train_pytorch.py            # PyTorch single model training
│   ├── train_ensemble.py           # 3-model ensemble training
│   ├── train_optimized.py          # 5-model optimized ensemble ⭐
│   ├── evaluate.py                 # Model evaluation
│   └── utils.py                    # Utility functions
├── models/                         # Trained models (see Releases)
├── plots/                          # Training history plots
├── results/                        # Evaluation results
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── classification_report.txt
├── requirements.txt                # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## 🏗️ Architecture

### Ensemble Models

1. **EfficientNet-B0 Variant** (60.05%)
   - Dropout: 0.35
   - Hidden layers: 384 → 192
   - Freeze point: 120 layers

2. **ResNet50** (70.50%) ⭐ Best Single Model
   - Dropout: 0.5 → 0.4 → 0.3
   - Hidden layers: 512 → 256
   - Freeze point: 100 layers

3. **DenseNet121** (66.03%)
   - Dropout: 0.5 → 0.4 → 0.3
   - Hidden layers: 512 → 256
   - Freeze point: 150 layers

4. **EfficientNet-B2** (64.94%)
   - Dropout: 0.5 → 0.45 → 0.35
   - Hidden layers: 768 → 384
   - Freeze point: 150 layers

### Training Configuration

```python
Optimizer: AdamW (lr=2e-3, weight_decay=0.02)
Scheduler: OneCycleLR (cosine annealing)
Loss: CrossEntropyLoss + class weights + label smoothing (0.1)
Batch Size: 32
Image Size: 69×69 (resized from 256×256 for memory)
Epochs: 50-60 with early stopping (patience=12)
Mixed Precision: Enabled (AMP)
```

## 📈 Results

### Model Performance

| Model | Val Acc | Parameters | Training Time |
|-------|---------|------------|---------------|
| EfficientNet-B0 | 60.05% | 5.3M | ~30 min |
| ResNet50 | 70.50% | 25.6M | ~45 min |
| DenseNet121 | 66.03% | 8.0M | ~35 min |
| EfficientNet-B2 | 64.94% | 9.2M | ~40 min |
| **Ensemble + TTA** | **76.14%** | **48.1M** | **~3 hours** |

### Class Distribution

The dataset has significant class imbalance:

```
Class 4 (Cigar Shaped): 334 images (1.9%) ← Most imbalanced
Class 2 (Round Smooth): 2,645 images (14.9%) ← Most common
```

Handled using balanced class weights.

## 🔬 Methodology

### Key Techniques

1. **Transfer Learning**: Pre-trained ImageNet weights
2. **Fine-tuning**: Unfreeze last 100-150 layers
3. **Class Balancing**: Compute balanced class weights
4. **Regularization**: Dropout + L2 + BatchNorm + Label Smoothing
5. **Ensemble Strategy**: Weighted voting by validation accuracy
6. **Test-Time Augmentation**: 5x horizontal flips

### Why This Approach Works

- **Transfer Learning**: Leverages ImageNet features (saves training time)
- **Ensemble Diversity**: Different architectures capture different patterns
- **TTA**: Reduces prediction variance, improves robustness
- **Class Weights**: Prevents bias toward majority classes

## 💻 Hardware Requirements

### Minimum
- CPU: Any modern processor
- RAM: 8GB
- GPU: None (CPU training supported, slower)

### Recommended (Used for this project)
- CPU: Intel i5/AMD Ryzen 5 or better
- RAM: 16GB
- GPU: **NVIDIA RTX 3050 (4GB VRAM)**
- CUDA: 12.1+

## 📊 Visualizations

The project includes comprehensive evaluation:
- **Confusion Matrix**: Per-class performance analysis
- **Sample Predictions**: Visual verification of model outputs
- **Training Curves**: Loss and accuracy progression
- **Classification Report**: Precision, recall, F1-score per class

## 🎓 Skills Demonstrated

This project showcases:
- ✅ Deep Learning (PyTorch)
- ✅ Transfer Learning & Fine-tuning
- ✅ Ensemble Methods
- ✅ Class Imbalance Handling
- ✅ GPU Optimization (CUDA, AMP)
- ✅ Hyperparameter Tuning
- ✅ Production ML Workflows
- ✅ Code Organization & Documentation
- ✅ Version Control (Git)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Galaxy10 DECals by [Bovy et al.](http://astro.utoronto.ca/~bovy/Galaxy10/)
- **Frameworks**: PyTorch, scikit-learn
- **Pre-trained Models**: ImageNet weights from torchvision

## 📧 Contact

**[Your Name]**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

**⭐ If you find this project helpful for your learning, please consider giving it a star!**

*Built with ❤️ for astronomy and deep learning*
