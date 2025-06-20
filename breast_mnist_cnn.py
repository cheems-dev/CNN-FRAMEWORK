"""
BreastMNIST CNN Classification
Dataset: https://medmnist.com/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
import gzip
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def download_breast_mnist():
    """Download BreastMNIST dataset"""
    print("Downloading BreastMNIST dataset...")
    
    # Create data directory
    data_dir = Path("data/breast_mnist")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for BreastMNIST dataset
    base_url = "https://zenodo.org/record/6496656/files/"
    files = {
        "breastmnist.npz": f"{base_url}breastmnist.npz"
    }
    
    try:
        for filename, url in files.items():
            filepath = data_dir / filename
            if not filepath.exists():
                print(f"Downloading {filename}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
        
        print(f"Dataset downloaded to {data_dir}")
        return data_dir
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def load_breast_mnist_data(data_dir):
    """Load BreastMNIST dataset"""
    npz_file = data_dir / "breastmnist.npz"
    
    if not npz_file.exists():
        print("Dataset file not found.")
        return None, None, None, None, None, None
    
    try:
        data = np.load(npz_file)
        
        # Extract data
        train_images = data['train_images']
        train_labels = data['train_labels']
        val_images = data['val_images']
        val_labels = data['val_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
        
        return train_images, train_labels, val_images, val_labels, test_images, test_labels
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None, None, None, None

def create_breast_mnist_cnn_model(input_shape, num_classes):
    """Create CNN model for BreastMNIST classification"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history, dataset_name):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title(f'{dataset_name} Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title(f'{dataset_name} Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test, dataset_name):
    """Evaluate model and return metrics"""
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Print results
    print(f"\n{dataset_name} CNN Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, precision, recall, f1

def main():
    print("Starting BreastMNIST CNN Classification...")
    
    # Download dataset
    data_dir = download_breast_mnist()
    if data_dir is None:
        print("Failed to download dataset. Exiting.")
        return None
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_breast_mnist_data(data_dir)
    if X_train is None:
        print("Failed to load dataset. Exiting.")
        return None
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Prepare data
    # Add channel dimension if missing (for grayscale images)
    if len(X_train.shape) == 3:
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    num_classes = len(np.unique(y_train))
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    
    # Create model
    model = create_breast_mnist_cnn_model(X_train.shape[1:], num_classes)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint('breast_mnist_best_model.h5', save_best_only=True)
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train_cat,
        batch_size=64,
        epochs=50,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, 'BreastMNIST')
    
    # Evaluate model
    accuracy, precision, recall, f1 = evaluate_model(
        model, X_test, y_test_cat, 'BreastMNIST'
    )
    
    # Save results
    results = {
        'Dataset': 'BreastMNIST',
        'Model': 'CNN',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('breast_mnist_results.csv', index=False)
    
    print("BreastMNIST CNN training completed!")
    return results

if __name__ == "__main__":
    main()