"""
Integration tests for API endpoints
Tests FastAPI and Flask servers
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestFastAPI:
    """Test FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        try:
            from api_server import app
            return TestClient(app)
        except ImportError:
            pytest.skip("API server not available")
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "device" in data
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint"""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_classes_endpoint(self, client):
        """Test classes listing endpoint"""
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert "count" in data
        assert data["count"] == 10
    
    @pytest.mark.skipif(
        not any(os.path.exists(f'models/orchestrated_{name}.pth') 
                for name in ['resnet50', 'densenet121', 'efficientnet_b0_v1']),
        reason="No trained models found"
    )
    def test_predict_endpoint(self, client):
        """Test prediction endpoint"""
        # Create dummy image
        img = Image.new('RGB', (256, 256), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Send request
        response = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "top3" in data
            assert "individual_models" in data
            assert "all_probabilities" in data
        else:
            # Models might not be loaded
            assert response.status_code in [500, 503]
    
    def test_predict_invalid_file(self, client):
        """Test prediction with invalid file"""
        # Send text file instead of image
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        assert response.status_code in [400, 500]
    
    def test_predict_no_file(self, client):
        """Test prediction without file"""
        response = client.post("/predict")
        assert response.status_code == 422  # Validation error


class TestFlaskApp:
    """Test Flask inference server"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        try:
            from inference_server import app
            app.config['TESTING'] = True
            return app.test_client()
        except ImportError:
            pytest.skip("Inference server not available")
    
    def test_index_page(self, client):
        """Test index page loads"""
        response = client.get('/')
        assert response.status_code == 200
    
    def test_health_endpoint(self, client):
        """Test health endpoint"""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert "status" in data
        assert "models_loaded" in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
