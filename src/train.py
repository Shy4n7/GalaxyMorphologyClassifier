"""
Training script for Galaxy10 classifier.
Loads data, builds model, trains with callbacks, and saves results.
"""

import os
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.append(os.path.dirname(__file__))

from load_data import load_galaxy10_data
from model import build_model
from utils import plot_training_history, check_gpu


def train_galaxy_classifier(
    model_type='efficientnet',
    epochs=100,
    batch_size=16,  # Optimized for RTX 3050 (4GB VRAM)
    learning_rate=1e-3
):
    """
    Train the Galaxy10 classifier with GPU optimization.
    
    Args:
        model_type (str): Type of model ('cnn' or 'efficientnet')
        epochs (int): Maximum number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
    """
    print("\n" + "=" * 80)
    print("GALAXY10 CLASSIFIER TRAINING (GPU OPTIMIZED)")
    print("=" * 80 + "\n")
    
    # Enable mixed precision for RTX 3050
    try:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled: float16 computation, float32 variables")
    except Exception as e:
        print(f"Mixed precision not available: {e}")
    
    # Check GPU availability
    gpu_available = check_gpu()
    
    # Configure GPU memory growth to avoid OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU configuration warning: {e}")
    
    # Load data
    print("Loading Galaxy10 dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data()
    
    # Calculate class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights calculated for balanced training:")
    for cls, weight in class_weight_dict.items():
        print(f"  Class {cls}: {weight:.2f}")
    
    # Build model
    model = build_model(
        model_type=model_type,
        input_shape=(69, 69, 3),
        num_classes=10,
        learning_rate=learning_rate
    )
    
    # Print model summary
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    model.summary()
    
    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{model_type}_{timestamp}"
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    callbacks_list = [
        # Early stopping: stop if validation loss doesn't improve for 5 epochs
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint: save best model
        keras.callbacks.ModelCheckpoint(
            filepath=f'models/best_model_{model_name}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=f'logs/{model_name}',
            histogram_freq=1,
            write_graph=True
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"Model type: {model_type}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"GPU available: {gpu_available}")
    print("=" * 80 + "\n")
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        class_weight=class_weight_dict,  # Handle class imbalance
        verbose=1
    )
    
    # Plot training history
    print("\nGenerating training plots...")
    plot_training_history(history, save_path=f'plots/training_history_{model_name}.png')
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save final model
    final_model_path = f'models/final_model_{model_name}.h5'
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save best model path for easy access
    best_model_path = f'models/best_model_{model_name}.h5'
    with open('models/latest_model.txt', 'w') as f:
        f.write(best_model_path)
    print(f"Best model path saved to: models/latest_model.txt")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nTo view TensorBoard logs, run:")
    print(f"  tensorboard --logdir=logs/{model_name}")
    print(f"\nTo evaluate the model, run:")
    print(f"  python src/evaluate.py")
    print("=" * 80 + "\n")
    
    return model, history


if __name__ == "__main__":
    # Configuration (Optimized for RTX 3050)
    MODEL_TYPE = 'efficientnet'  # Options: 'cnn' or 'efficientnet'
    EPOCHS = 100  # More epochs with GPU
    BATCH_SIZE = 16  # Optimized for 4GB VRAM
    LEARNING_RATE = 1e-3
    
    # Train model
    model, history = train_galaxy_classifier(
        model_type=MODEL_TYPE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
