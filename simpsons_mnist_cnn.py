#!/usr/bin/env python3
"""
SimpsonsMNIST CNN Classification
Dataset: https://github.com/alvarobartt/simpsons-mnist
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
import zipfile
import io
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def download_simpsons_dataset():
    """Download and extract SimpsonsMNIST dataset"""
    print("Downloading SimpsonsMNIST dataset...")
    
    # Create data directory
    data_dir = Path("data/simpsons")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    url = "https://github.com/alvarobartt/simpsons-mnist/raw/master/data/simpsons_mnist.zip"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Extract zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            zip_file.extractall(data_dir)
        
        print(f"Dataset downloaded and extracted to {data_dir}")
        return data_dir
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def load_simpsons_data(data_dir):
    """Load SimpsonsMNIST dataset"""
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        print("Dataset directories not found. Trying alternative structure...")
        # Try to find the actual data structure
        for root, dirs, files in os.walk(data_dir):
            print(f"Found directory: {root}")
            print(f"Subdirectories: {dirs}")
            print(f"Files: {files[:5]}...")  # Show first 5 files
            if len(files) > 0:
                break
        return None, None, None, None
    
    # Load training data
    train_images = []
    train_labels = []
    
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for img_path in class_dir.glob("*.png"):
                img = Image.open(img_path).convert('RGB')
                img = img.resize((64, 64))  # Resize to standard size
                train_images.append(np.array(img))
                train_labels.append(class_name)
    
    # Load test data
    test_images = []
    test_labels = []
    
    for class_dir in test_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for img_path in class_dir.glob("*.png"):
                img = Image.open(img_path).convert('RGB')
                img = img.resize((64, 64))
                test_images.append(np.array(img))
                test_labels.append(class_name)
    
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

def create_simpsons_cnn_model(input_shape, num_classes):
    """Create CNN model for SimpsonsMNIST classification"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    plt.tight_layout()
    plt.savefig('simpsons_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model and return metrics"""
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Print results
    print(f"\nSimpsonsMNIST CNN Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('SimpsonsMNIST Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('simpsons_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, precision, recall, f1

def main():
    print("Starting SimpsonsMNIST CNN Classification...")
    
    # Download dataset
    data_dir = download_simpsons_dataset()
    if data_dir is None:
        print("Failed to download dataset. Exiting.")
        return
    
    # Load data
    X_train, y_train, X_test, y_test = load_simpsons_data(data_dir)
    if X_train is None:
        print("Failed to load dataset. Exiting.")
        return
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Classes: {np.unique(y_train)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Convert to categorical
    num_classes = len(label_encoder.classes_)
    y_train_cat = keras.utils.to_categorical(y_train_encoded, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test_encoded, num_classes)
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Create model
    model = create_simpsons_cnn_model(X_train.shape[1:], num_classes)
    
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
        keras.callbacks.ModelCheckpoint('simpsons_best_model.h5', save_best_only=True)
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train_cat,
        batch_size=32,
        epochs=50,
        validation_data=(X_test, y_test_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    accuracy, precision, recall, f1 = evaluate_model(
        model, X_test, y_test_cat, label_encoder.classes_
    )
    
    # Save results
    results = {
        'Dataset': 'SimpsonsMNIST',
        'Model': 'CNN',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('simpsons_results.csv', index=False)
    
    print("SimpsonsMNIST CNN training completed!")
    return results

if __name__ == "__main__":
    main()