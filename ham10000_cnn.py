"""
HAM10000 CNN Classification
Dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
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
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def download_ham10000_dataset():
    """Download HAM10000 dataset (metadata and images)"""
    print("Downloading HAM10000 dataset...")
    
    # Create data directory
    data_dir = Path("data/ham10000")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for HAM10000 dataset
    base_url = "https://dataverse.harvard.edu/api/access/datafile/"
    files = {
        "HAM10000_metadata.csv": f"{base_url}3324987",
        "HAM10000_images_part_1.zip": f"{base_url}3324986",
        "HAM10000_images_part_2.zip": f"{base_url}3324985"
    }
    
    try:
        # Download metadata
        csv_path = data_dir / "HAM10000_metadata.csv"
        if not csv_path.exists():
            print("Downloading metadata...")
            response = requests.get(files["HAM10000_metadata.csv"])
            response.raise_for_status()
            with open(csv_path, 'wb') as f:
                f.write(response.content)
        
        # Download and extract images
        for part in [1, 2]:
            zip_filename = f"HAM10000_images_part_{part}.zip"
            zip_path = data_dir / zip_filename
            
            if not zip_path.exists():
                print(f"Downloading {zip_filename}...")
                response = requests.get(files[zip_filename], stream=True)
                response.raise_for_status()
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Extract images
            images_dir = data_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            print(f"Extracting {zip_filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(images_dir)
        
        print(f"Dataset downloaded and extracted to {data_dir}")
        return data_dir
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please manually download the dataset from:")
        print("https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        return None

def load_ham10000_data(data_dir, image_size=(224, 224), sample_size=None):
    """Load HAM10000 dataset"""
    csv_path = data_dir / "HAM10000_metadata.csv"
    images_dir = data_dir / "images"
    
    if not csv_path.exists():
        print("Metadata CSV not found.")
        return None, None
    
    # Load metadata
    df = pd.read_csv(csv_path)
    
    # If sample_size is specified, take a random sample
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Using sample of {sample_size} images")
    
    print(f"Loading {len(df)} images...")
    print(f"Class distribution:")
    print(df['dx'].value_counts())
    
    # Load images
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        image_id = row['image_id']
        label = row['dx']
        
        # Try different image extensions
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = images_dir / f"{image_id}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        
        if image_path and image_path.exists():
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize(image_size)
                images.append(np.array(img))
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {image_id}: {e}")
                continue
        else:
            print(f"Image not found: {image_id}")
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"Loaded {idx + 1}/{len(df)} images")
    
    print(f"Successfully loaded {len(images)} images")
    return np.array(images), np.array(labels)

def create_ham10000_cnn_model(input_shape, num_classes):
    """Create CNN model for HAM10000 classification"""
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
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fifth convolutional block
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
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

def evaluate_model(model, X_test, y_test, class_names, dataset_name):
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
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, precision, recall, f1

def main():
    print("Starting HAM10000 CNN Classification...")
    
    # Download dataset
    data_dir = download_ham10000_dataset()
    if data_dir is None:
        print("Failed to download dataset. Exiting.")
        return None
    
    # Load data (using a sample for faster processing)
    X, y = load_ham10000_data(data_dir, sample_size=2000)  # Use sample for demo
    if X is None:
        print("Failed to load dataset. Exiting.")
        return None
    
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Convert to categorical
    num_classes = len(label_encoder.classes_)
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Create model
    model = create_ham10000_cnn_model(X_train.shape[1:], num_classes)
    
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
        keras.callbacks.ModelCheckpoint('ham10000_best_model.h5', save_best_only=True)
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train_cat,
        batch_size=32,
        epochs=50,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, 'HAM10000')
    
    # Evaluate model
    accuracy, precision, recall, f1 = evaluate_model(
        model, X_test, y_test_cat, label_encoder.classes_, 'HAM10000'
    )
    
    # Save results
    results = {
        'Dataset': 'HAM10000',
        'Model': 'CNN',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('ham10000_results.csv', index=False)
    
    print("HAM10000 CNN training completed!")
    return results

if __name__ == "__main__":
    main()