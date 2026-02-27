import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_dataset, prepare_data

# Configuration
DATASET_PATH = 'Dataset'
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 50

def build_custom_cnn(input_shape, num_classes):
    """Build custom CNN model"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_transfer_model(base_model_name, input_shape, num_classes):
    """Build transfer learning model"""
    try:
        if base_model_name == 'VGG16':
            base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'ResNet50':
            base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'MobileNetV2':
            base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    except Exception as e:
        print(f"Error loading {base_model_name}: {e}")
        raise
    
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

def train_model(model, model_name, X_train, y_train, X_val, y_val):
    """Train model with callbacks and augmentation"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        horizontal_flip=True
    )
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(f'models/{model_name}', save_best_only=True, monitor='val_accuracy', mode='max', save_format='tf')
    ]
    
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val / 255.0, y_val),
        epochs=EPOCHS,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    return history

def evaluate_model(model, model_name, X_test, y_test, class_names):
    """Evaluate model and save metrics"""
    y_pred = np.argmax(model.predict(X_test / 255.0), axis=1)
    
    print(f"\n{model_name} Results:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_confusion_matrix.png')
    plt.close()
    
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    return accuracy

def main():
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load and prepare data
    print("Loading dataset...")
    images, labels, class_names = load_dataset(DATASET_PATH, IMG_SIZE)
    print(f"Loaded {len(images)} images from {len(class_names)} classes: {class_names}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(images, labels)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    input_shape = (*IMG_SIZE, 3)
    num_classes = len(class_names)
    
    # Train models
    models_config = [
        ('CustomCNN', lambda: build_custom_cnn(input_shape, num_classes)),
        ('VGG16', lambda: build_transfer_model('VGG16', input_shape, num_classes)),
        ('ResNet50', lambda: build_transfer_model('ResNet50', input_shape, num_classes)),
        ('MobileNetV2', lambda: build_transfer_model('MobileNetV2', input_shape, num_classes))
    ]
    
    results = {}
    for model_name, model_builder in models_config:
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}")
        
        model = model_builder()
        history = train_model(model, model_name, X_train, y_train, X_val, y_val)
        accuracy = evaluate_model(model, model_name, X_test, y_test, class_names)
        results[model_name] = accuracy
        
        print(f"{model_name} Test Accuracy: {accuracy*100:.2f}%")
    
    # Save results summary
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name}: {acc*100:.2f}%")
    
    # Save class names
    np.save('models/class_names.npy', class_names)
    print("\nTraining complete! Models saved in 'models/' directory")

if __name__ == '__main__':
    main()
