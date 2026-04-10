"""
Grad-CAM (Gradient-weighted Class Activation Mapping) utilities.
Visualizes which parts of the galaxy image the model focuses on for classification.
"""

import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn.functional as F


def get_target_layer(model, model_name):
    """
    Get the appropriate target layer for Grad-CAM based on model architecture.
    
    Args:
        model: PyTorch model
        model_name (str): Name of the model architecture
        
    Returns:
        target_layer: Layer to compute gradients from
    """
    if 'resnet' in model_name.lower():
        # ResNet: use last convolutional layer
        return [model.resnet.layer4[-1]]
    elif 'densenet' in model_name.lower():
        # DenseNet: use last dense block
        return [model.densenet.features[-1]]
    elif 'efficientnet' in model_name.lower():
        # EfficientNet: use last convolutional layer
        return [model.efficientnet.features[-1]]
    else:
        # Default: try to find last conv layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return [module]
        raise ValueError(f"Could not find target layer for {model_name}")


def generate_gradcam(model, image_tensor, predicted_class, model_name='resnet'):
    """
    Generate Grad-CAM visualization for a given image and predicted class.
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor (1, 3, H, W)
        predicted_class (int): Predicted class index
        model_name (str): Model architecture name
        
    Returns:
        numpy.ndarray: Grad-CAM heatmap overlay on original image (H, W, 3)
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get target layer
    target_layers = get_target_layer(model, model_name)
    
    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Specify target class
    targets = [ClassifierOutputTarget(predicted_class)]
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    
    # Get the CAM for the first image in batch
    grayscale_cam = grayscale_cam[0, :]
    
    # Convert input tensor to numpy for visualization
    # Denormalize if needed (assuming ImageNet normalization)
    image_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    # Normalize to [0, 1] if not already
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    
    # Ensure values are in [0, 1]
    image_np = np.clip(image_np, 0, 1)
    
    # Create visualization
    visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
    
    return visualization


def generate_ensemble_gradcam(models, model_names, image_tensor, predicted_class):
    """
    Generate Grad-CAM visualizations for all models in the ensemble.
    
    Args:
        models (list): List of PyTorch models
        model_names (list): List of model names
        image_tensor: Input image tensor (1, 3, H, W)
        predicted_class (int): Predicted class index
        
    Returns:
        dict: Dictionary mapping model names to Grad-CAM visualizations
    """
    visualizations = {}
    
    for model, name in zip(models, model_names):
        try:
            vis = generate_gradcam(model, image_tensor, predicted_class, name)
            visualizations[name] = vis
        except Exception as e:
            print(f"Warning: Could not generate Grad-CAM for {name}: {e}")
            # Return original image if Grad-CAM fails
            image_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            image_np = np.clip(image_np, 0, 1)
            visualizations[name] = (image_np * 255).astype(np.uint8)
    
    return visualizations


def create_gradcam_grid(visualizations, original_image):
    """
    Create a grid showing original image + Grad-CAM from all models.
    
    Args:
        visualizations (dict): Dictionary of model name -> Grad-CAM image
        original_image: Original input image
        
    Returns:
        numpy.ndarray: Grid image
    """
    # Prepare original image
    if isinstance(original_image, torch.Tensor):
        orig_np = original_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        orig_np = np.clip(orig_np, 0, 1)
        orig_np = (orig_np * 255).astype(np.uint8)
    else:
        orig_np = original_image
    
    # Resize all images to same size
    target_size = (224, 224)
    orig_resized = cv2.resize(orig_np, target_size)
    
    # Create list of images: [original, model1_cam, model2_cam, ...]
    images = [orig_resized]
    labels = ['Original']
    
    for name, vis in visualizations.items():
        vis_resized = cv2.resize(vis, target_size)
        images.append(vis_resized)
        labels.append(name)
    
    # Create grid (2 rows if more than 3 images)
    n_images = len(images)
    if n_images <= 3:
        rows, cols = 1, n_images
    else:
        cols = 3
        rows = (n_images + cols - 1) // cols
    
    # Add padding
    padding = 10
    label_height = 30
    
    # Calculate grid size
    grid_height = rows * (target_size[0] + label_height + padding) + padding
    grid_width = cols * (target_size[1] + padding) + padding
    
    # Create white canvas
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    # Place images
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        
        y_start = row * (target_size[0] + label_height + padding) + padding
        x_start = col * (target_size[1] + padding) + padding
        
        # Place image
        grid[y_start:y_start+target_size[0], x_start:x_start+target_size[1]] = img
        
        # Add label
        label_y = y_start + target_size[0] + 20
        cv2.putText(grid, label, (x_start, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return grid


if __name__ == "__main__":
    # Test Grad-CAM with a sample model
    print("Grad-CAM utilities loaded successfully!")
    print("Use generate_gradcam() to create visualizations")
