"""
Unit tests for Galaxy Classifier
Tests data loading, model architecture, and inference
"""

import os
import sys
import pytest
import torch
import numpy as np
from PIL import Image
import io

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from load_data import load_galaxy10_data, get_class_names

class TestDataLoading:
    """Test data loading functionality"""
    
    def test_class_names_count(self):
        """Test that we have exactly 10 galaxy classes"""
        class_names = get_class_names()
        assert len(class_names) == 10, "Should have 10 galaxy classes"
    
    def test_class_names_not_empty(self):
        """Test that class names are not empty strings"""
        class_names = get_class_names()
        for name in class_names:
            assert len(name) > 0, "Class names should not be empty"
            assert isinstance(name, str), "Class names should be strings"

class TestModelArchitecture:
    """Test model architectures"""
    
    def test_convnext_model_import(self):
        """Test that ConvNeXt model can be imported"""
        try:
            from train_optimized_v3 import build_convnext_tiny
            model = build_convnext_tiny(num_classes=10)
            assert model is not None, "ConvNeXt model should be created"
        except ImportError:
            pytest.skip("ConvNeXt model not available")
    
    def test_densenet_model_import(self):
        """Test that DenseNet model can be imported"""
        try:
            from train_optimized_v3 import build_densenet161
            model = build_densenet161(num_classes=10)
            assert model is not None, "DenseNet model should be created"
        except ImportError:
            pytest.skip("DenseNet model not available")
    
    def test_resnext_model_import(self):
        """Test that ResNeXt model can be imported"""
        try:
            from train_optimized_v3 import build_resnext50
            model = build_resnext50(num_classes=10)
            assert model is not None, "ResNeXt model should be created"
        except ImportError:
            pytest.skip("ResNeXt model not available")
    
    def test_model_output_shape(self):
        """Test that model outputs correct shape"""
        try:
            from train_optimized_v3 import build_resnext50
            model = build_resnext50(num_classes=10)
            model.eval()
            
            # Create dummy input
            x = torch.randn(1, 3, 224, 224)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 10), "Output should be (batch_size, num_classes)"
            
        except ImportError:
            pytest.skip("Model not available")
    
    def test_model_forward_pass(self):
        """Test that model can perform forward pass"""
        try:
            from train_optimized_v3 import build_resnext50
            model = build_resnext50(num_classes=10)
            model.eval()
            
            # Create dummy input
            x = torch.randn(2, 3, 224, 224)
            
            with torch.no_grad():
                output = model(x)
            
            # Check output is valid
            assert not torch.isnan(output).any(), "Output should not contain NaN"
            assert not torch.isinf(output).any(), "Output should not contain Inf"
            
        except ImportError:
            pytest.skip("Model not available")

class TestInference:
    """Test inference functionality"""
    
    def test_image_preprocessing(self):
        """Test image preprocessing pipeline"""
        try:
            from inference_server import preprocess_image
            
            # Create dummy image
            img = Image.new('RGB', (256, 256), color='red')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Preprocess
            tensor = preprocess_image(img_bytes.read())
            
            # Check shape
            assert tensor.shape == (1, 3, 224, 224), "Preprocessed image should be (1, 3, 224, 224)"
            
        except ImportError:
            pytest.skip("Inference module not available")

class TestAPI:
    """Test API functionality"""
    
    def test_api_import(self):
        """Test that API server can be imported"""
        try:
            import inference_server
            assert hasattr(inference_server, 'app'), "API should have Flask app"
        except ImportError:
            pytest.skip("API server not available")
    
    def test_class_names_constant(self):
        """Test that CLASS_NAMES is defined correctly"""
        try:
            from inference_server import CLASS_NAMES
            assert len(CLASS_NAMES) == 10, "Should have 10 class names"
            assert all(isinstance(name, str) for name in CLASS_NAMES), "All names should be strings"
        except ImportError:
            pytest.skip("API server not available")

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
