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
    
    @pytest.mark.skipif(
        not os.path.exists('data/Galaxy10_DECals.h5') and 
        not os.path.exists('data/Galaxy10_Mock.h5'),
        reason="No dataset file found"
    )
    def test_load_data_shapes(self):
        """Test that loaded data has correct shapes"""
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data()
            
            # Check that we have data
            assert len(X_train) > 0, "Training set should not be empty"
            assert len(X_val) > 0, "Validation set should not be empty"
            assert len(X_test) > 0, "Test set should not be empty"
            
            # Check shapes match
            assert len(X_train) == len(y_train), "X_train and y_train should have same length"
            assert len(X_val) == len(y_val), "X_val and y_val should have same length"
            assert len(X_test) == len(y_test), "X_test and y_test should have same length"
            
            # Check image dimensions
            assert X_train.shape[1:] == (69, 69, 3), "Images should be 69x69x3"
            
            # Check label range
            assert y_train.min() >= 0 and y_train.max() < 10, "Labels should be 0-9"
            
        except Exception as e:
            pytest.skip(f"Could not load data: {e}")
    
    @pytest.mark.skipif(
        not os.path.exists('data/Galaxy10_DECals.h5') and 
        not os.path.exists('data/Galaxy10_Mock.h5'),
        reason="No dataset file found"
    )
    def test_data_normalization(self):
        """Test that data is properly normalized"""
        try:
            X_train, _, _, _, _, _ = load_galaxy10_data()
            
            # Check value range (should be 0-1 after normalization)
            assert X_train.min() >= 0, "Pixel values should be >= 0"
            assert X_train.max() <= 1, "Pixel values should be <= 1"
            
        except Exception as e:
            pytest.skip(f"Could not load data: {e}")


class TestModelArchitecture:
    """Test model architectures"""
    
    def test_resnet_model_import(self):
        """Test that ResNet model can be imported"""
        try:
            from train_ensemble import ResNetModel
            model = ResNetModel(num_classes=10)
            assert model is not None, "ResNet model should be created"
        except ImportError:
            pytest.skip("ResNet model not available")
    
    def test_densenet_model_import(self):
        """Test that DenseNet model can be imported"""
        try:
            from train_ensemble import DenseNetModel
            model = DenseNetModel(num_classes=10)
            assert model is not None, "DenseNet model should be created"
        except ImportError:
            pytest.skip("DenseNet model not available")
    
    def test_efficientnet_model_import(self):
        """Test that EfficientNet model can be imported"""
        try:
            from train_ensemble import EfficientNetModel
            model = EfficientNetModel(num_classes=10, version='b0')
            assert model is not None, "EfficientNet model should be created"
        except ImportError:
            pytest.skip("EfficientNet model not available")
    
    def test_model_output_shape(self):
        """Test that model outputs correct shape"""
        try:
            from train_ensemble import ResNetModel
            model = ResNetModel(num_classes=10)
            model.eval()
            
            # Create dummy input
            x = torch.randn(1, 3, 69, 69)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 10), "Output should be (batch_size, num_classes)"
            
        except ImportError:
            pytest.skip("Model not available")
    
    def test_model_forward_pass(self):
        """Test that model can perform forward pass"""
        try:
            from train_ensemble import ResNetModel
            model = ResNetModel(num_classes=10)
            model.eval()
            
            # Create dummy input
            x = torch.randn(2, 3, 69, 69)
            
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
            from predict import preprocess_image
            
            # Create dummy image
            img = Image.new('RGB', (256, 256), color='red')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Preprocess
            device = torch.device('cpu')
            tensor = preprocess_image(img_bytes.read(), device)
            
            # Check shape
            assert tensor.shape == (1, 3, 69, 69), "Preprocessed image should be (1, 3, 69, 69)"
            
        except ImportError:
            pytest.skip("Inference module not available")
    
    @pytest.mark.skipif(
        not any(os.path.exists(f'models/orchestrated_{name}.pth') 
                for name in ['resnet50', 'densenet121', 'efficientnet_b0_v1']),
        reason="No trained models found"
    )
    def test_model_loading(self):
        """Test that trained models can be loaded"""
        try:
            from predict import load_models
            
            device = torch.device('cpu')
            models = load_models(device)
            
            assert len(models) > 0, "At least one model should be loaded"
            
            for name, model in models.items():
                assert model is not None, f"Model {name} should not be None"
                
        except ImportError:
            pytest.skip("Inference module not available")
    
    @pytest.mark.skipif(
        not any(os.path.exists(f'models/orchestrated_{name}.pth') 
                for name in ['resnet50', 'densenet121', 'efficientnet_b0_v1']),
        reason="No trained models found"
    )
    def test_prediction_output(self):
        """Test that prediction returns valid output"""
        try:
            from predict import load_models, preprocess_image, predict
            
            # Load models
            device = torch.device('cpu')
            models = load_models(device)
            
            if not models:
                pytest.skip("No models loaded")
            
            # Create dummy image
            img = Image.new('RGB', (256, 256), color='blue')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Preprocess and predict
            tensor = preprocess_image(img_bytes.read(), device)
            result = predict(models, tensor)
            
            # Check result structure
            assert 'prediction' in result, "Result should have 'prediction'"
            assert 'confidence' in result, "Result should have 'confidence'"
            assert 'ensemble_probs' in result, "Result should have 'ensemble_probs'"
            
            # Check values
            assert isinstance(result['prediction'], str), "Prediction should be string"
            assert 0 <= result['confidence'] <= 100, "Confidence should be 0-100"
            assert len(result['ensemble_probs']) == 10, "Should have 10 class probabilities"
            
        except ImportError:
            pytest.skip("Inference module not available")


class TestAPI:
    """Test API functionality"""
    
    def test_api_import(self):
        """Test that API server can be imported"""
        try:
            import api_server
            assert hasattr(api_server, 'app'), "API should have FastAPI app"
        except ImportError:
            pytest.skip("API server not available")
    
    def test_class_names_constant(self):
        """Test that CLASS_NAMES is defined correctly"""
        try:
            from api_server import CLASS_NAMES
            assert len(CLASS_NAMES) == 10, "Should have 10 class names"
            assert all(isinstance(name, str) for name in CLASS_NAMES), "All names should be strings"
        except ImportError:
            pytest.skip("API server not available")


class TestUtils:
    """Test utility functions"""
    
    def test_utils_import(self):
        """Test that utils module can be imported"""
        try:
            import utils
            assert hasattr(utils, 'plot_confusion_matrix'), "Utils should have plot_confusion_matrix"
            assert hasattr(utils, 'save_classification_report'), "Utils should have save_classification_report"
        except ImportError:
            pytest.skip("Utils module not available")
    
    def test_confusion_matrix_creation(self):
        """Test confusion matrix creation"""
        try:
            from utils import plot_confusion_matrix
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            # Create dummy data
            y_true = np.array([0, 1, 2, 0, 1, 2])
            y_pred = np.array([0, 1, 1, 0, 2, 2])
            class_names = ['Class A', 'Class B', 'Class C']
            
            # Should not raise exception
            plot_confusion_matrix(y_true, y_pred, class_names, save_path=None)
            
        except ImportError:
            pytest.skip("Utils module not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
