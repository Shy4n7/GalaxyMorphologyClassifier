# Galaxy Morphology Classifier - Frontend Requirements

## 🎯 Project Overview

Build a modern web application frontend for a Deep Learning Galaxy Morphology Classifier. The backend is a Flask API running on `http://localhost:5001` with an ensemble of 4 CNN models achieving **76.14% accuracy** on galaxy classification.

---

## 🔌 Backend API Endpoints

### 1. **POST /api/predict**
Classify a galaxy image using the ensemble model.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file upload)

**Response:**
```json
{
  "prediction": "Barred Spiral",
  "confidence": 76.14,
  "top3": [
    {"class": "Barred Spiral", "confidence": 76.14},
    {"class": "Unbarred Tight Spiral", "confidence": 12.3},
    {"class": "Round Smooth", "confidence": 5.2}
  ],
  "individual_models": {
    "resnet50": {"class": "Barred Spiral", "confidence": 78.5},
    "densenet121": {"class": "Barred Spiral", "confidence": 74.2},
    "efficientnet_b0_v1": {"class": "Unbarred Tight Spiral", "confidence": 72.1},
    "efficientnet_b0_v2": {"class": "Barred Spiral", "confidence": 75.8}
  },
  "all_probabilities": {
    "Disturbed Galaxies": 2.1,
    "Merging Galaxies": 1.8,
    "Round Smooth": 5.2,
    "In-between Round Smooth": 0.8,
    "Cigar Shaped Smooth": 1.2,
    "Barred Spiral": 76.14,
    "Unbarred Tight Spiral": 12.3,
    "Unbarred Loose Spiral": 0.9,
    "Edge-on without Bulge": 0.3,
    "Edge-on with Bulge": 0.2
  }
}
```

### 2. **POST /api/gradcam**
Generate Grad-CAM visualization showing which parts of the image the models focus on.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file upload)

**Response:**
```json
{
  "grid": "base64_encoded_image_string",
  "predicted_class": "Barred Spiral",
  "confidence": 76.14
}
```

The `grid` is a 2x2 grid showing Grad-CAM heatmaps for all 4 models overlaid on the original image.

### 3. **GET /api/health**
Check if the server and models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 4
}
```

---

## 🌌 Galaxy Classification System

### **10 Galaxy Classes:**
1. **Disturbed Galaxies** - Irregular, chaotic morphology
2. **Merging Galaxies** - Two or more galaxies colliding
3. **Round Smooth** - Elliptical, smooth appearance
4. **In-between Round Smooth** - Transitional elliptical
5. **Cigar Shaped Smooth** - Elongated elliptical
6. **Barred Spiral** - Spiral with central bar structure
7. **Unbarred Tight Spiral** - Tightly wound spiral arms
8. **Unbarred Loose Spiral** - Loosely wound spiral arms
9. **Edge-on without Bulge** - Disk galaxy viewed edge-on, no bulge
10. **Edge-on with Bulge** - Disk galaxy viewed edge-on, with bulge

### **4 Ensemble Models:**

| Model | Parameters | Accuracy | Architecture |
|-------|-----------|----------|--------------|
| ResNet-50 | 25.6M | 74.2% | Deep Residual Learning |
| DenseNet-121 | 8.0M | 73.8% | Dense Connections |
| EfficientNet-B0 V1 | 5.3M | 72.5% | Efficient Scaling |
| EfficientNet-B0 V2 | 5.3M | 73.1% | Efficient Scaling |

**Ensemble Method:** Probability averaging (soft voting)  
**Overall Ensemble Accuracy:** 76.14%

---

## ✨ Required Frontend Features

### **1. Image Upload & Classification**

**Must Have:**
- Drag-and-drop file upload or file picker
- Image preview before classification
- "Classify" button to trigger prediction
- Loading state during API call
- Display ensemble prediction with confidence percentage
- Show top-3 predictions with confidence bars
- Display all 10 class probabilities (bar chart or list)

**Nice to Have:**
- Sample galaxy images for quick testing (5-10 samples)
- Image preprocessing info display (69x69 resize, RGB format)
- Upload history/recent classifications

---

### **2. Individual Model Predictions**

**Must Have:**
- Display all 4 model predictions separately
- Show each model's predicted class and confidence
- Visual indicators (cards, panels, or grid layout)
- Model metadata (name, architecture, accuracy, parameters)

**Nice to Have:**
- Model comparison view
- Highlight when models disagree
- Model performance statistics

---

### **3. Grad-CAM Visualization**

**Must Have:**
- "Show Grad-CAM" button after classification
- Display the 2x2 grid image showing attention maps for all 4 models
- Label each quadrant with model name
- Show which regions the models focus on

**Nice to Have:**
- Toggle between original image and Grad-CAM overlay
- Zoom/pan functionality
- Download Grad-CAM image
- Individual model Grad-CAM views

---

### **4. Results Display**

**Must Have:**
- Clear, prominent display of ensemble prediction
- Confidence percentage (large, easy to read)
- Top-3 predictions with visual confidence bars
- Galaxy class name and description

**Nice to Have:**
- Probability distribution chart (all 10 classes)
- Comparison with individual model predictions
- Explanation of ensemble voting
- Galaxy type information/facts

---

### **5. User Interface**

**Must Have:**
- Clean, modern design
- Responsive layout (desktop + mobile)
- Clear navigation
- Loading states and error handling
- Accessible color contrast

**Suggested Theme:**
- Dark mode (space/astronomy inspired)
- Primary color: Blue (#135bec)
- Accent colors: Purple, Emerald, Amber
- Glassmorphism or neumorphism effects
- Smooth animations

**Nice to Have:**
- Multiple view modes (simple, detailed, expert)
- Settings panel
- Light/dark mode toggle
- Keyboard shortcuts

---

## 🎨 Suggested UI Layout

### **Main View:**

```
┌─────────────────────────────────────────────────────────┐
│  Header: Galaxy Morphology Classifier                   │
├──────────────────┬──────────────────────────────────────┤
│                  │                                      │
│  Upload Area     │   Results Panel                      │
│  ┌────────────┐  │   ┌──────────────────────────────┐  │
│  │            │  │   │ Ensemble Prediction:         │  │
│  │   Image    │  │   │ Barred Spiral - 76.14%       │  │
│  │  Preview   │  │   └──────────────────────────────┘  │
│  │            │  │                                      │
│  └────────────┘  │   Top 3 Predictions:                │
│                  │   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 76.14%            │
│  [Classify]      │   ▓▓▓▓▓ 12.3%                       │
│  [Grad-CAM]      │   ▓▓ 5.2%                           │
│                  │                                      │
│  Sample Images:  │   Individual Models:                │
│  [🌌][🌌][🌌]    │   ┌──────┬──────┬──────┬──────┐    │
│  [🌌][🌌]        │   │ResNet│Dense │Eff-V1│Eff-V2│    │
│                  │   │78.5% │74.2% │72.1% │75.8% │    │
│                  │   └──────┴──────┴──────┴──────┘    │
└──────────────────┴──────────────────────────────────────┘
```

---

## 📊 Data Visualization Ideas

1. **Confidence Bars** - Horizontal bars for top-3 predictions
2. **Probability Chart** - Bar chart or pie chart for all 10 classes
3. **Model Comparison** - Side-by-side cards showing each model's prediction
4. **Grad-CAM Grid** - 2x2 heatmap visualization
5. **Ensemble Voting** - Visual representation of how models voted

---

## 🔧 Technical Specifications

### **Image Requirements:**
- Input: Any image format (JPEG, PNG)
- Backend preprocessing: Resize to 69x69, RGB format
- Frontend: Show original image, send as-is to backend

### **API Integration:**
- Base URL: `http://localhost:5001`
- CORS: Enabled on backend
- File upload: `multipart/form-data`
- Response: JSON

### **Performance:**
- Classification time: ~1-2 seconds
- Grad-CAM generation: ~2-3 seconds
- Show loading indicators during API calls

---

## 🎯 User Flow

1. **User uploads galaxy image** (drag-drop or file picker)
2. **Image preview** displays
3. **User clicks "Classify"**
4. **Loading state** shows
5. **Results appear:**
   - Ensemble prediction (large, prominent)
   - Top-3 predictions with bars
   - Individual model predictions
6. **User clicks "Show Grad-CAM"** (optional)
7. **Grad-CAM visualization** displays
8. **User can upload another image** or try samples

---

## 🌟 Bonus Features (Optional)

1. **Batch Classification** - Upload multiple images
2. **Export Results** - Download predictions as JSON/CSV
3. **Model Comparison Mode** - Compare predictions across models
4. **Galaxy Information** - Educational content about galaxy types
5. **Statistics Dashboard** - Show classification statistics
6. **Confidence Threshold** - Filter predictions by confidence
7. **API Documentation** - Interactive API docs in the UI
8. **Dark/Light Theme** - Theme switcher

---

## 📝 Sample Images Available

The backend has 10 sample galaxy images (one for each class):
- `sample_1_Disturbed.jpg`
- `sample_2_Merging.jpg`
- `sample_3_Round_Smooth.jpg`
- `sample_4_In_between_Round_Smooth.jpg`
- `sample_5_Cigar_Shaped_Smooth.jpg`
- `sample_6_Barred_Spiral.jpg`
- `sample_7_Unbarred_Tight_Spiral.jpg`
- `sample_8_Unbarred_Loose_Spiral.jpg`
- `sample_9_Edge_on_without_Bulge.jpg`
- `sample_10_Edge_on_with_Bulge.jpg`

These can be served from the backend at `/static/samples/` or copied to the frontend.

---

## 🚀 Technology Suggestions

**Frontend Framework:**
- React, Vue, or Svelte
- Or vanilla HTML/CSS/JavaScript

**Styling:**
- Tailwind CSS, Bootstrap, or custom CSS
- Chart library: Chart.js, Recharts, or D3.js

**State Management:**
- React Context, Zustand, or Redux (if using React)

**Build Tool:**
- Vite, Webpack, or Parcel

---

## ✅ Success Criteria

A successful frontend should:
1. ✅ Allow users to upload galaxy images easily
2. ✅ Display ensemble prediction clearly and prominently
3. ✅ Show individual model predictions
4. ✅ Visualize Grad-CAM attention maps
5. ✅ Handle loading and error states gracefully
6. ✅ Be responsive and accessible
7. ✅ Have a clean, modern, space-themed design
8. ✅ Provide a smooth, intuitive user experience

---

## 📞 Backend Contact

- **API Base URL:** `http://localhost:5001`
- **Health Check:** `GET /api/health`
- **Classification:** `POST /api/predict`
- **Grad-CAM:** `POST /api/gradcam`

---

**Good luck building an amazing Galaxy Classifier frontend! 🌌✨**
