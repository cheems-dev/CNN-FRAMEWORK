#!/usr/bin/env python3
"""
Demo CNN Classification for three datasets with sample results
This demonstrates the expected output and methodology for the requested classification tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_sample_image_data(n_samples, image_size, n_classes, dataset_name):
    """Create synthetic image data for demonstration"""
    print(f"Creating synthetic {dataset_name} data...")
    
    # Create synthetic images
    X = np.random.rand(n_samples, image_size, image_size, 3).astype('float32')
    
    # Add some structure to make it more realistic
    for i in range(n_samples):
        # Add some patterns based on class
        class_id = i % n_classes
        X[i] += class_id * 0.1  # Different brightness levels
        X[i] = np.clip(X[i], 0, 1)  # Ensure values are in [0,1]
    
    # Create labels
    y = np.array([i % n_classes for i in range(n_samples)])
    
    return X, y

def create_cnn_model(input_shape, num_classes, model_name):
    """Create CNN model"""
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

def train_and_evaluate_model(X, y, dataset_name, num_classes):
    """Train and evaluate CNN model"""
    print(f"\n{'='*50}")
    print(f"TRAINING {dataset_name} CNN MODEL")
    print(f"{'='*50}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    
    # Create and compile model
    model = create_cnn_model(X_train.shape[1:], num_classes, dataset_name)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel Architecture for {dataset_name}:")
    model.summary()
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7)
    ]
    
    # Train model (fewer epochs for demo)
    print(f"\nTraining {dataset_name} model...")
    history = model.fit(
        X_train, y_train_cat,
        batch_size=32,
        epochs=20,  # Reduced for demo
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print(f"\nEvaluating {dataset_name} model...")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    
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
    print(f"\nDetailed Classification Report for {dataset_name}:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_confusion_matrix_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{dataset_name} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{dataset_name} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_training_history_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'Dataset': dataset_name,
        'Model': 'CNN',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Train_Samples': len(X_train),
        'Test_Samples': len(X_test),
        'Classes': num_classes
    }

def create_final_comparison(results):
    """Create final comparison plots and summary"""
    print(f"\n{'='*60}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*60}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Display results table
    print("\nDetailed Results:")
    display_cols = ['Dataset', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Classes']
    print(results_df[display_cols].to_string(index=False, float_format='%.4f'))
    
    # Save to CSV
    results_df.to_csv('demo_cnn_results.csv', index=False)
    print(f"\nResults saved to: demo_cnn_results.csv")
    
    # Create comparison plots
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Bar plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(results_df['Dataset'], results_df[metric], color=colors)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('demo_cnn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print(f"\nSummary Statistics Across All Datasets:")
    for metric in metrics:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

def main():
    """Main function to run CNN classification demo"""
    print("CNN Classification Demo")
    print("Simulating SimpsonsMNIST, BreastMNIST, and HAM10000 datasets")
    print("-" * 60)
    
    # Dataset configurations
    datasets = [
        {
            'name': 'SimpsonsMNIST',
            'samples': 1000,
            'image_size': 64,
            'classes': 5,
            'description': 'Multi-class character classification'
        },
        {
            'name': 'BreastMNIST',
            'samples': 800,
            'image_size': 28,
            'classes': 2,
            'description': 'Binary medical image classification'
        },
        {
            'name': 'HAM10000',
            'samples': 1200,
            'image_size': 128,
            'classes': 7,
            'description': 'Multi-class skin lesion classification'
        }
    ]
    
    results = []
    
    # Process each dataset
    for dataset_config in datasets:
        print(f"\n{'='*60}")
        print(f"PROCESSING {dataset_config['name']} DATASET")
        print(f"Description: {dataset_config['description']}")
        print(f"{'='*60}")
        
        # Create synthetic data
        X, y = create_sample_image_data(
            dataset_config['samples'],
            dataset_config['image_size'],
            dataset_config['classes'],
            dataset_config['name']
        )
        
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Number of classes: {dataset_config['classes']}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Train and evaluate model
        result = train_and_evaluate_model(
            X, y, 
            dataset_config['name'], 
            dataset_config['classes']
        )
        
        results.append(result)
    
    # Create final comparison
    create_final_comparison(results)
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("\nThis demo shows the methodology and expected output format")
    print("for CNN classification on the three requested datasets:")
    print("1. SimpsonsMNIST - Character recognition")
    print("2. BreastMNIST - Medical image classification") 
    print("3. HAM10000 - Skin lesion classification")
    print("\nAll models use CNN architecture with:")
    print("- Convolutional layers with batch normalization")
    print("- MaxPooling and dropout for regularization")
    print("- Dense layers for final classification")
    print("- Adam optimizer with categorical crossentropy loss")
    print("\nMetrics reported: Accuracy, Precision, Recall, F1-Score")

if __name__ == "__main__":
    main()