// DOM refs
const fileInput       = document.getElementById('file-input');
const dropZone        = document.getElementById('drop-zone');
const previewImage    = document.getElementById('preview-image');
const uploadContent   = document.getElementById('upload-content');
const scanLine        = document.getElementById('scan-line');
const randomBtn       = document.getElementById('random-btn');
const classifyBtn     = document.getElementById('classify-btn');
const validationPanel = document.getElementById('validation-panel');
const groundTruthLabel= document.getElementById('ground-truth-label');
const validationResult= document.getElementById('validation-result');
const sampleInfoPanel = document.getElementById('sample-info-panel');
const sampleOriginalImage = document.getElementById('sample-original-image');
const sampleClass     = document.getElementById('sample-class');
const sampleSource    = document.getElementById('sample-source');
const sampleType      = document.getElementById('sample-type');
const top3Section     = document.getElementById('top3-section');
const top3Bars        = document.getElementById('top3-bars');

const MODEL_COLORS = {
    convnext:    '#ef4444',
    resnext50:   '#10b981',
    densenet161: '#f97316',
};

let selectedFile = null;
let randomManifest = [];
let isProcessing = false;
let currentSampleData = null;

// Load manifest
fetch('/samples/random_manifest.json')
    .then(r => r.json())
    .then(d => { randomManifest = d; })
    .catch(() => {});

// Events
randomBtn.addEventListener('click', handleRandom);
classifyBtn.addEventListener('click', handleClassify);
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => handleFileSelect(e.target.files[0]));
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = '#135bec'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = 'rgba(19,91,236,0.3)'; });
dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.style.borderColor = 'rgba(19,91,236,0.3)';
    if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
});

function handleFileSelect(file, sampleData = null) {
    if (!file) return;
    selectedFile = file;
    currentSampleData = sampleData;

    const reader = new FileReader();
    reader.onload = e => {
        previewImage.src = e.target.result;
        previewImage.hidden = false;
        uploadContent.hidden = true;
        classifyBtn.disabled = false;
        resetResults();

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
    const item = randomManifest[Math.floor(Math.random() * randomManifest.length)];
    try {
        const blob = await fetch(item.src).then(r => r.blob());
        handleFileSelect(new File([blob], 'random.jpg', { type: 'image/jpeg' }), item);
    } catch {}
}

async function handleClassify() {
    if (!selectedFile || isProcessing) return;
    isProcessing = true;
    classifyBtn.disabled = true;
    scanLine.hidden = false;
    classifyBtn.innerHTML = '<span class="material-icons">sync</span> Processing...';

    try {
        const fd = new FormData();
        fd.append('image', selectedFile);
        const res = await fetch('/api/predict', { method: 'POST', body: fd });
        const data = await res.json();
        displayResults(data);
    } catch {
        alert('Prediction failed. Is the server running?');
    } finally {
        isProcessing = false;
        classifyBtn.disabled = false;
        scanLine.hidden = true;
        classifyBtn.innerHTML = '<span class="material-icons">play_arrow</span> Run Ensemble';
    }
}

function displayResults(data) {
    // Confidence circle
    document.getElementById('confidence-value').textContent = data.confidence.toFixed(1);
    document.querySelector('.circle').style.strokeDasharray = `${data.confidence}, 100`;

    // Prediction label
    document.getElementById('prediction-label').textContent = data.prediction;

    // Model cards
    const models = { convnext: 'convnext', resnext50: 'resnext50', densenet161: 'densenet161' };
    for (const [id] of Object.entries(models)) {
        const result = data.individual_models[id];
        if (!result) continue;

        const predEl  = document.getElementById(`lab-pred-${id}`);
        const scoreEl = document.getElementById(`lab-score-${id}`);
        const barEl   = document.getElementById(`bar-${id}`);

        if (predEl)  predEl.textContent  = result.class;
        if (scoreEl) scoreEl.textContent = `${result.confidence.toFixed(1)}%`;
        if (barEl)   barEl.style.width   = `${result.confidence}%`;
    }

    // Top-3 predictions
    if (data.top3 && data.top3.length) {
        top3Section.hidden = false;
        const colors = ['#135bec', '#7c3aed', '#475569'];
        top3Bars.innerHTML = data.top3.map((item, i) => `
            <div class="top3-bar-row">
                <div class="top3-label-row">
                    <span class="top3-name">${item.class}</span>
                    <span class="top3-pct">${item.confidence.toFixed(1)}%</span>
                </div>
                <div class="top3-track">
                    <div class="top3-fill" style="width:${item.confidence}%;background:${colors[i]}"></div>
                </div>
            </div>
        `).join('');
    }

    // Validation
    if (currentSampleData && currentSampleData.label) {
        validationPanel.hidden = false;
        groundTruthLabel.textContent = currentSampleData.label;
        const correct = data.prediction === currentSampleData.label;
        validationResult.textContent = correct ? '✓ CORRECT' : '✗ INCORRECT';
        validationResult.className = `validation-result ${correct ? 'correct' : 'incorrect'}`;
    } else {
        validationPanel.hidden = true;
    }
}

function resetResults() {
    document.getElementById('confidence-value').textContent = '0';
    document.getElementById('prediction-label').textContent = 'AWAITING INPUT';
    document.querySelector('.circle').style.strokeDasharray = '0, 100';

    ['convnext', 'resnext50', 'densenet161'].forEach(id => {
        const pred  = document.getElementById(`lab-pred-${id}`);
        const score = document.getElementById(`lab-score-${id}`);
        const bar   = document.getElementById(`bar-${id}`);
        if (pred)  pred.textContent  = '---';
        if (score) score.textContent = '---';
        if (bar)   bar.style.width   = '0%';
    });

    top3Section.hidden = true;
    top3Bars.innerHTML = '';
    validationPanel.hidden = true;
}

// Particles
if (window.particlesJS) {
    particlesJS('particles-js', {
        particles: {
            number: { value: 60 },
            color: { value: '#135bec' },
            shape: { type: 'circle' },
            opacity: { value: 0.4, random: true },
            size: { value: 2, random: true },
            line_linked: { enable: true, distance: 140, color: '#135bec', opacity: 0.2, width: 1 },
            move: { enable: true, speed: 1.2 },
        },
        interactivity: {
            events: { onhover: { enable: true, mode: 'grab' } },
            modes: { grab: { distance: 120, line_linked: { opacity: 0.5 } } },
        },
    });
}

// GPU stats
async function updateGPUUsage() {
    try {
        const data = await fetch('/api/gpu-stats').then(r => r.json());
        const el = document.getElementById('gpu-usage');
        if (!el) return;
        if (data.available) {
            el.textContent = `${data.utilization_percent}%`;
            el.style.color = data.utilization_percent > 80 ? '#ef4444' : '#10b981';
        } else {
            el.textContent = 'CPU';
            el.style.color = '#94a3b8';
        }
    } catch {}
}
setInterval(updateGPUUsage, 3000);
updateGPUUsage();
