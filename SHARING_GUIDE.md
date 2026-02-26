# Galaxy Classifier - Setup Guide for Friends 🌌

Hey! Here is how you can get my Galaxy Morphology Classifier running on your machine.

## 🚀 Quick Setup

### 1. Clone the Project
Open your terminal and run:
```bash
git clone https://github.com/Shy4n7/GalaxyMorphologyClassifier.git
cd GalaxyMorphologyClassifier
```

### 2. Install Dependencies
Make sure you have Python 3.11+ installed.
```bash
# Create a virtual environment
python -m venv .venv
# Activate it (Windows)
.venv\Scripts\activate
# Activate it (Mac/Linux)
# source .venv/bin/activate

# Install the essentials
pip install -r requirements.txt
```

### 3. Get the "Working Models"
The model weights are too large for Git, so you need to download them manually:
1. Go to the [v1.0.0 Release Assets](https://github.com/Shy4n7/GalaxyMorphologyClassifier/releases/tag/v1.0.0) page.
2. Download all **5** `.pth` files:
    - `orchestrated_resnet50.pth` (**Star-Lord**)
    - `orchestrated_densenet121.pth` (**Gamora**)
    - `orchestrated_efficientnet_b0_v1.pth` (**Drax**)
    - `orchestrated_efficientnet_b0_v2.pth` (**Rocket**)
    - `mobilenet_overfit_hard_acc98.00.pth` (**Groot**)
3. Place them inside the `models/` folder in the project directory.

---

## 🎮 How to Use

### Option A: The Beautiful Web App (Recommended)
Run this command to start the interactive interface:
```bash
python src/inference_server.py
```
Then open `http://localhost:8080` in your browser. You can drag and drop any galaxy image!

### Option B: The Pro Dashboard
If you want to see the system stats and the "Guardians" ensemble in action:
```bash
# Terminal 1: Start the backend
python src/dashboard_server.py

# Terminal 2: Start the frontend (requires Node.js)
cd frontend
npm install
npm run dev
```

---

## 🧠 What's under the hood?
This uses a weighted ensemble of 5 deep learning models (nicknamed the **Guardians of the Galaxy**) to achieve **95.4% accuracy**.

Enjoy exploring the cosmos! 🪐
