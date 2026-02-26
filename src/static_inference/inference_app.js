// Galaxy Classifier Inference App

const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const removeBtn = document.getElementById('remove-btn');
const classifyBtn = document.getElementById('classify-btn');
const resultsSection = document.getElementById('results-section');
const statusText = document.getElementById('status-text');

let selectedFile = null;

// Upload area click
uploadArea.addEventListener('click', () => {
    if (!selectedFile) {
        fileInput.click();
    }
});

// File selection
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

// Handle file
function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        document.querySelector('.upload-content').style.display = 'none';
        previewContainer.style.display = 'block';
        classifyBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Remove image
removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = '';
    previewContainer.style.display = 'none';
    document.querySelector('.upload-content').style.display = 'block';
    classifyBtn.disabled = true;
    resultsSection.style.display = 'none';
});

// Classify button
classifyBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Update UI
    classifyBtn.disabled = true;
    document.querySelector('.btn-text').style.display = 'none';
    document.querySelector('.btn-loader').style.display = 'inline-block';
    statusText.textContent = 'Classifying...';

    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        displayResults(result);
        statusText.textContent = 'Complete';

    } catch (error) {
        console.error('Error:', error);
        alert('Classification failed. Please try again.');
        statusText.textContent = 'Error';
    } finally {
        classifyBtn.disabled = false;
        document.querySelector('.btn-text').style.display = 'inline';
        document.querySelector('.btn-loader').style.display = 'none';
    }
});

// Display results
function displayResults(result) {
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Main prediction
    document.getElementById('prediction-class').textContent = result.prediction;
    document.getElementById('confidence-text').textContent = `${result.confidence.toFixed(1)}%`;
    document.getElementById('confidence-fill').style.width = `${result.confidence}%`;

    // Top 3 predictions
    const top3List = document.getElementById('top3-list');
    top3List.innerHTML = result.top3.map((pred, index) => `
        <div class="prediction-item">
            <span class="class-name">${index + 1}. ${pred.class}</span>
            <span class="confidence">${pred.confidence.toFixed(1)}%</span>
        </div>
    `).join('');

    // Individual models
    const modelsGrid = document.getElementById('models-grid');
    modelsGrid.innerHTML = Object.entries(result.individual_models).map(([name, pred]) => `
        <div class="model-card">
            <div class="model-name">${formatModelName(name)}</div>
            <div class="model-prediction">${truncateText(pred.class, 20)}</div>
            <div class="model-confidence">${pred.confidence.toFixed(1)}%</div>
        </div>
    `).join('');

    // Probability distribution
    const chartBars = document.getElementById('chart-bars');
    const sortedProbs = Object.entries(result.all_probabilities)
        .sort((a, b) => b[1] - a[1]);

    chartBars.innerHTML = sortedProbs.map(([className, prob]) => `
        <div class="chart-bar">
            <div class="bar-label">${truncateText(className, 25)}</div>
            <div class="bar-container">
                <div class="bar-fill" style="width: ${prob}%"></div>
            </div>
            <div class="bar-value">${prob.toFixed(1)}%</div>
        </div>
    `).join('');
}

// Grad-CAM visualization
async function showGradCAM() {
    if (!selectedFile) return;

    const gradcamBtn = document.getElementById('gradcam-btn');
    const gradcamSection = document.getElementById('gradcam-section');

    // Update button state
    gradcamBtn.disabled = true;
    gradcamBtn.textContent = 'Generating...';

    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/api/gradcam', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Grad-CAM generation failed');
        }

        const result = await response.json();

        // Display Grad-CAM grid
        const gradcamImage = document.getElementById('gradcam-image');
        gradcamImage.src = `data:image/png;base64,${result.grid}`;

        // Show section
        gradcamSection.style.display = 'block';
        gradcamSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Update button
        gradcamBtn.textContent = 'Visualization Generated';

    } catch (error) {
        console.error('Error:', error);
        alert('Grad-CAM generation failed. Please try again.');
        gradcamBtn.textContent = 'Show What AI Sees';
        gradcamBtn.disabled = false;
    }
}

// Helper functions
function formatModelName(name) {
    const nameMap = {
        'resnet50': 'ResNet-50',
        'densenet121': 'DenseNet-121',
        'efficientnet_b0_v1': 'EfficientNet-B0 V1',
        'efficientnet_b0_v2': 'EfficientNet-B0 V2',
        'efficientnet_b2': 'EfficientNet-B2'
    };
    return nameMap[name] || name;
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength - 3) + '...';
}

// Check server health on load
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('Server health:', data);
        statusText.textContent = `Ready (${data.models_loaded} models)`;
    } catch (error) {
        console.error('Health check failed:', error);
        statusText.textContent = 'Server Error';
    }
}

// Load sample image
async function loadSample(imagePath, label) {
    try {
        const response = await fetch(imagePath);
        const blob = await response.blob();
        const file = new File([blob], imagePath.split('/').pop(), { type: 'image/jpeg' });
        handleFile(file);

        // Scroll to upload section
        document.querySelector('.upload-section').scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (error) {
        console.error('Error loading sample:', error);
        alert('Failed to load sample image');
    }
}

// Initialize
checkHealth();
