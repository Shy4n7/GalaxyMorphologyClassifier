import os
import sys
import pytest
import io
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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
    
    def test_health_endpoint(self, client):
        """Test health endpoint"""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert "status" in data
        assert "models_loaded" in data
        assert "device" in data

    def test_gpu_stats_endpoint(self, client):
        """Test gpu-stats endpoint"""
        response = client.get('/api/gpu-stats')
        assert response.status_code == 200
        data = response.get_json()
        assert "available" in data

    def test_predict_no_file(self, client):
        """Test prediction without file"""
        response = client.post("/api/predict")
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_predict_invalid_file(self, client):
        """Test prediction with empty filename"""
        response = client.post(
            "/api/predict",
            data={"image": (io.BytesIO(b""), "")}
        )
        assert response.status_code == 400

    def test_gradcam_no_file(self, client):
        response = client.post("/api/gradcam")
        assert response.status_code == 400

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
