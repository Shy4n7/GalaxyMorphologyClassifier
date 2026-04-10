// Galaxy Classifier API Service

const API_BASE_URL = 'http://localhost:5001';

export interface PredictionResponse {
    prediction: string;
    confidence: number;
    top3: Array<{
        class: string;
        confidence: number;
    }>;
    individual_models: {
        [key: string]: {
            class: string;
            confidence: number;
        };
    };
    all_probabilities: {
        [key: string]: number;
    };
}

export interface GradCAMResponse {
    grid: string; // base64 encoded image
    predicted_class: string;
    confidence: number;
}

export async function classifyGalaxy(imageFile: File): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error('Classification failed');
    }

    return await response.json();
}

export async function getGradCAM(imageFile: File): Promise<GradCAMResponse> {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${API_BASE_URL}/api/gradcam`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error('Grad-CAM generation failed');
    }

    return await response.json();
}

export async function checkHealth(): Promise<{ status: string; models_loaded: number }> {
    const response = await fetch(`${API_BASE_URL}/api/health`);

    if (!response.ok) {
        throw new Error('Health check failed');
    }

    return await response.json();
}
