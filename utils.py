import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_dataset(dataset_path, img_size=(224, 224)):
    """Load images and labels from dataset directory"""
    images, labels = [], []
    class_names = sorted(os.listdir(dataset_path))
    
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                images.append(np.array(img))
                labels.append(idx)
            except:
                print(f"Error loading: {img_path}")
    
    return np.array(images), np.array(labels), class_names

def prepare_data(images, labels, test_size=0.2, val_size=0.1):
    """Split data into train, validation, and test sets"""
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=(test_size + val_size), random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=42, stratify=y_temp
    )
    
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_augmentation():
    """Create data augmentation layer"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])
