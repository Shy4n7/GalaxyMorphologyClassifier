"""
Utility functions for plotting and visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os


def plot_training_history(history, save_path='plots/training_history.png'):
    """
    Plot training and validation accuracy and loss curves.
    
    Args:
        history: Keras History object from model.fit()
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/confusion_matrix.png'):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names (list): List of class names
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def save_classification_report(y_true, y_pred, class_names, save_path='results/classification_report.txt'):
    """
    Generate and save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names (list): List of class names
        save_path (str): Path to save the report
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Galaxy10 Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
        f.write("\n\n")
        
        # Add per-class accuracy
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        f.write("Per-Class Accuracy:\n")
        f.write("-" * 80 + "\n")
        for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
            f.write(f"{i}: {name:45s} - {acc:.4f}\n")
    
    print(f"Classification report saved to {save_path}")
    
    # Also print to console
    print("\n" + "=" * 80)
    print(report)
    print("\nPer-Class Accuracy:")
    print("-" * 80)
    for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
        print(f"{i}: {name:45s} - {acc:.4f}")


def plot_sample_predictions(X_test, y_true, y_pred, class_names, num_samples=12, save_path='results/sample_predictions.png'):
    """
    Plot random sample predictions with images, true labels, and predicted labels.
    
    Args:
        X_test: Test images
        y_true: True labels
        y_pred: Predicted labels
        class_names (list): List of class names
        num_samples (int): Number of samples to display
        save_path (str): Path to save the plot
    """
    # Randomly select samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    # Create subplot grid
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Display image
        axes[i].imshow(X_test[idx])
        axes[i].axis('off')
        
        # Get labels
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        
        # Color: green if correct, red if incorrect
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        
        # Set title
        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label}",
            fontsize=9,
            color=color,
            fontweight='bold'
        )
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sample predictions saved to {save_path}")
    plt.close()


def check_gpu():
    """
    Check if GPU is available and print device information.
    """
    import tensorflow as tf
    
    print("=" * 80)
    print("GPU DETECTION")
    print("=" * 80)
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✓ GPU Available: {len(gpus)} device(s) detected")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        # Print additional GPU info
        print(f"\nTensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"GPU support: {tf.test.is_gpu_available()}")
    else:
        print("✗ No GPU detected. Training will use CPU (slower).")
        print("  To enable GPU, install tensorflow-gpu and CUDA toolkit.")
    
    print("=" * 80 + "\n")
    
    return len(gpus) > 0
