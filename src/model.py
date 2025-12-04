"""
Model architecture module for Galaxy10 classification.
Provides both custom CNN and EfficientNetB0 transfer learning options.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def build_custom_cnn(input_shape=(69, 69, 3), num_classes=10, learning_rate=1e-3):
    """
    Build a custom CNN architecture for galaxy classification.
    
    Args:
        input_shape (tuple): Input image shape (default: (69, 69, 3))
        num_classes (int): Number of output classes (default: 10)
        learning_rate (float): Learning rate for Adam optimizer
        
    Returns:
        keras.Model: Compiled model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Data augmentation layer
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_efficientnet(input_shape=(69, 69, 3), num_classes=10, learning_rate=1e-3):
    """
    Build improved EfficientNetB0 transfer learning model for galaxy classification.
    Optimized for RTX 3050 with better regularization.
    
    Args:
        input_shape (tuple): Input image shape (default: (69, 69, 3))
        num_classes (int): Number of output classes (default: 10)
        learning_rate (float): Learning rate for Adam optimizer
        
    Returns:
        keras.Model: Compiled model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.3)(x)  # Increased rotation
    x = layers.RandomZoom(0.2)(x)  # Increased zoom
    x = layers.RandomContrast(0.2)(x)  # Added contrast augmentation
    
    # Load pre-trained EfficientNetB0 (without top layers)
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Fine-tune more layers for better performance
    base_model.trainable = True
    # Freeze first 100 layers, train the rest
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Build improved classification head
    x = base_model(x, training=True)
    x = layers.Dropout(0.4)(x)  # Increased dropout
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)  # float32 for stability
    
    model = models.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_model(model_type='efficientnet', input_shape=(69, 69, 3), num_classes=10, learning_rate=1e-3):
    """
    Build and return a compiled model based on the specified type.
    
    Args:
        model_type (str): Type of model ('cnn' or 'efficientnet')
        input_shape (tuple): Input image shape
        num_classes (int): Number of output classes
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        keras.Model: Compiled model
    """
    if model_type.lower() == 'cnn':
        print("Building Custom CNN model...")
        model = build_custom_cnn(input_shape, num_classes, learning_rate)
    elif model_type.lower() == 'efficientnet':
        print("Building EfficientNetB0 transfer learning model...")
        model = build_efficientnet(input_shape, num_classes, learning_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'cnn' or 'efficientnet'")
    
    print(f"\nModel built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    return model


if __name__ == "__main__":
    # Test model building
    print("Testing Custom CNN:")
    cnn_model = build_model('cnn')
    cnn_model.summary()
    
    print("\n" + "="*80 + "\n")
    
    print("Testing EfficientNetB0:")
    eff_model = build_model('efficientnet')
    eff_model.summary()
