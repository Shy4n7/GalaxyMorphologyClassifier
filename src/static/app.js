
// DOM Elements
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const previewImage = document.getElementById('preview-image');
const uploadContent = document.getElementById('upload-content');
const scanLine = document.getElementById('scan-line');
const randomBtn = document.getElementById('random-btn');
const classifyBtn = document.getElementById('classify-btn');
const validationPanel = document.getElementById('validation-panel');
const groundTruthLabel = document.getElementById('ground-truth-label');
const validationResult = document.getElementById('validation-result');
const sampleInfoPanel = document.getElementById('sample-info-panel');
const sampleOriginalImage = document.getElementById('sample-original-image');
const sampleClass = document.getElementById('sample-class');
const sampleSource = document.getElementById('sample-source');
const sampleType = document.getElementById('sample-type');

// State
let selectedFile = null;
let randomManifest = [];
let isProcessing = false;
let currentSampleData = null; // Store complete sample data

// Load Manifest
fetch('/samples/random_manifest.json')
    .then(res => res.json())
    .then(data => {
        randomManifest = data;
        console.log("Manifest loaded:", data.length);
    })
    .catch(err => console.log("Manifest error:", err));

// Event Listeners
randomBtn.addEventListener('click', handleRandom);
classifyBtn.addEventListener('click', handleClassify);

// Drag & Drop
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#135bec';
});
dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'rgba(19, 91, 236, 0.3)';
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'rgba(19, 91, 236, 0.3)';
    if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
});

function handleFileSelect(file, sampleData = null) {
    if (!file) return;
    selectedFile = file;
    currentSampleData = sampleData; // Store complete sample data

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.hidden = false;
        uploadContent.hidden = true;
        classifyBtn.disabled = false;

        // Reset Results
        resetResults();

        // Show sample info if available
        if (currentSampleData) {
            sampleInfoPanel.hidden = false;
            sampleOriginalImage.src = currentSampleData.src;
            sampleClass.textContent = currentSampleData.label || 'Unknown';
            sampleSource.textContent = currentSampleData.src.split('/').pop();
            sampleType.textContent = currentSampleData.type || 'Unknown';
        } else {
            sampleInfoPanel.hidden = true;
        }
    };
    reader.readAsDataURL(file);
}


async function handleRandom() {
    if (!randomManifest.length) return;

    const randomItem = randomManifest[Math.floor(Math.random() * randomManifest.length)];
    // Fetch image blob
    try {
        const res = await fetch(randomItem.src);
        const blob = await res.blob();
        const file = new File([blob], "random.jpg", { type: "image/jpeg" });

        // Pass complete sample data to handleFileSelect
        handleFileSelect(file, randomItem);
    } catch (e) {
        console.error("Random error", e);
    }
}

async function handleClassify() {
    if (!selectedFile || isProcessing) return;

    isProcessing = true;
    classifyBtn.disabled = true;
    scanLine.hidden = false;
    classifyBtn.innerHTML = '<span class="material-icons">sync</span> PROCESSING...';

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        displayResults(data);
    } catch (e) {
        console.error("Prediction failed", e);
        alert("Prediction failed. Is the server running?");
    } finally {
        isProcessing = false;
        classifyBtn.disabled = false;
        scanLine.hidden = true;
        classifyBtn.innerHTML = '<span class="material-icons">play_arrow</span> RUN ENSEMBLE';
    }
}

function displayResults(data) {
    // Top Result
    document.getElementById('confidence-value').textContent = data.confidence.toFixed(1);
    document.getElementById('prediction-label').textContent = data.prediction;

    // Circle Animation
    const circle = document.querySelector('.circle');
    const offset = 100 - data.confidence;
    circle.style.strokeDasharray = `${data.confidence}, 100`;

    // Lab Grid Mapping — 3-model ensemble
    const models = {
        'convnext': {
            scoreId: 'lab-score-convnext',
            predId: 'lab-pred-convnext'
        },
        'resnext50': {
            scoreId: 'lab-score-resnext50',
            predId: 'lab-pred-resnext50'
        },
        'densenet161': {
            scoreId: 'lab-score-densenet161',
            predId: 'lab-pred-densenet161'
        }
    };

    for (const [id, config] of Object.entries(models)) {
        const result = data.individual_models[id];
        if (result) {
            // Update confidence score
            const scoreEl = document.getElementById(config.scoreId);
            if (scoreEl) {
                scoreEl.textContent = `${result.confidence.toFixed(1)}%`;
                scoreEl.style.color = result.confidence > 90 ? '#10b981' : 'inherit';
            }

            // Update prediction label
            const predEl = document.getElementById(config.predId);
            if (predEl) {
                predEl.textContent = result.class || '---';
            }
        }
    }

    // Validation (if sample data with ground truth is available)
    if (currentSampleData && currentSampleData.label) {
        validationPanel.hidden = false;
        groundTruthLabel.textContent = currentSampleData.label;

        const isCorrect = data.prediction === currentSampleData.label;
        validationResult.textContent = isCorrect ? '✓ CORRECT' : '✗ INCORRECT';
        validationResult.className = isCorrect ? 'validation-result correct' : 'validation-result incorrect';
    } else {
        validationPanel.hidden = true;
    }
}

function resetResults() {
    document.getElementById('confidence-value').textContent = '0';
    document.getElementById('prediction-label').textContent = 'AWAITING INPUT';
    document.querySelector('.circle').style.strokeDasharray = '0, 100';

    ['lab-score-convnext', 'lab-score-resnext50', 'lab-score-densenet161',
        'lab-pred-convnext', 'lab-pred-resnext50', 'lab-pred-densenet161']
        .forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = '---';
        });

    // Hide validation panel when resetting
    validationPanel.hidden = true;
    groundTruthLabel.textContent = '---';
    validationResult.textContent = '---';
    validationResult.className = 'validation-result';
}

// Particles Init
if (window.particlesJS) {
    particlesJS("particles-js", {
        "particles": {
            "number": { "value": 80 },
            "color": { "value": "#135bec" },
            "shape": { "type": "circle" },
            "opacity": { "value": 0.5 },
            "size": { "value": 3 },
            "line_linked": {
                "enable": true,
                "distance": 150,
                "color": "#135bec",
                "opacity": 0.4,
                "width": 1
            },
            "move": { "enable": true, "speed": 2 }
        }
    });
}

// GPU Usage Monitor
async function updateGPUUsage() {
    try {
        const res = await fetch('/api/gpu-stats');
        const data = await res.json();
        const gpuUsageEl = document.getElementById('gpu-usage');

        if (data.available) {
            gpuUsageEl.textContent = `${data.utilization_percent}%`;
            gpuUsageEl.style.color = data.utilization_percent > 80 ? '#ef4444' : '#10b981';
        } else {
            gpuUsageEl.textContent = 'CPU MODE';
            gpuUsageEl.style.color = '#94a3b8';
        }
    } catch (e) {
        console.error('GPU stats error:', e);
    }
}

// Update GPU usage every 2 seconds
setInterval(updateGPUUsage, 2000);
updateGPUUsage(); // Initial call
