"""
Evaluation script for Galaxy10 classifier.
Loads trained model and generates comprehensive evaluation metrics.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.append(os.path.dirname(__file__))

from load_data import load_galaxy10_data, get_class_names
from utils import (
    plot_confusion_matrix,
    save_classification_report,
    plot_sample_predictions,
    check_gpu
)


def evaluate_galaxy_classifier(model_path=None):
    """
    Evaluate the trained Galaxy10 classifier.
    
    Args:
        model_path (str): Path to saved model. If None, loads latest model.
    """
    print("\n" + "=" * 80)
    print("GALAXY10 CLASSIFIER EVALUATION")
    print("=" * 80 + "\n")
    
    # Check GPU
    check_gpu()
    
    # Load data
    print("Loading Galaxy10 dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data()
    class_names = get_class_names()
    
    # Load model
    if model_path is None:
        # Try to load latest model
        if os.path.exists('models/latest_model.txt'):
            with open('models/latest_model.txt', 'r') as f:
                model_path = f.read().strip()
            print(f"Loading latest model: {model_path}")
        else:
            raise FileNotFoundError(
                "No model path provided and models/latest_model.txt not found. "
                "Please train a model first using: python src/train.py"
            )
    
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    model.summary()
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Generate and save confusion matrix
    print("\n" + "=" * 80)
    print("GENERATING CONFUSION MATRIX")
    print("=" * 80)
    plot_confusion_matrix(
        y_test, y_pred, class_names,
        save_path='results/confusion_matrix.png'
    )
    
    # Generate and save classification report
    print("\n" + "=" * 80)
    print("GENERATING CLASSIFICATION REPORT")
    print("=" * 80)
    save_classification_report(
        y_test, y_pred, class_names,
        save_path='results/classification_report.txt'
    )
    
    # Generate and save sample predictions
    print("\n" + "=" * 80)
    print("GENERATING SAMPLE PREDICTIONS")
    print("=" * 80)
    plot_sample_predictions(
        X_test, y_test, y_pred, class_names,
        num_samples=12,
        save_path='results/sample_predictions.png'
    )
    
    # Additional metrics
    print("\n" + "=" * 80)
    print("ADDITIONAL METRICS")
    print("=" * 80)
    
    # Calculate per-class metrics
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nPer-Class Statistics:")
    print("-" * 80)
    print(f"{'Class':<50} {'Samples':>10} {'Accuracy':>10}")
    print("-" * 80)
    
    for i, name in enumerate(class_names):
        class_samples = np.sum(y_test == i)
        class_accuracy = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"{name:<50} {class_samples:>10} {class_accuracy:>10.4f}")
    
    print("-" * 80)
    print(f"{'Overall':<50} {len(y_test):>10} {test_accuracy:>10.4f}")
    
    # Top-3 and Top-5 accuracy
    top3_accuracy = np.mean([y_test[i] in np.argsort(y_pred_probs[i])[-3:] for i in range(len(y_test))])
    top5_accuracy = np.mean([y_test[i] in np.argsort(y_pred_probs[i])[-5:] for i in range(len(y_test))])
    
    print(f"\nTop-3 Accuracy: {top3_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - results/confusion_matrix.png")
    print("  - results/classification_report.txt")
    print("  - results/sample_predictions.png")
    print("=" * 80 + "\n")
    
    return model, y_pred, y_pred_probs


if __name__ == "__main__":
    # Evaluate model
    model, predictions, probabilities = evaluate_galaxy_classifier()
