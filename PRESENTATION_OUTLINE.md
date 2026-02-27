# Ophthalmology Image Classification - Presentation Outline

## Slide 1: Title
**Ophthalmology Image Classification: OCP vs Normal Eyes**
- Binary Classification using Deep Learning
- Your Name
- Date

## Slide 2: Problem Statement
- **Objective**: Automated classification of eye images
- **Classes**: 
  - A. OCP (Ocular Cicatricial Pemphigoid)
  - C. Normal Eyes
- **Challenge**: Small dataset (285 images total)

## Slide 3: Dataset Overview
- **Total Images**: 285
  - OCP: 148 images
  - Normal Eyes: 137 images
- **Split**: 70% Train, 10% Val, 20% Test
- **Image Size**: 256x256 pixels
- **Format**: JPG, TIF

## Slide 4: Methodology
**Models Tested**:
1. Custom CNN (6 convolutional layers)
2. VGG16 (Transfer Learning)
3. ResNet50 (Transfer Learning)
4. MobileNetV2 (Transfer Learning)

**Data Augmentation**:
- Rotation (10°)
- Horizontal flip
- Normalization (0-1)

## Slide 5: Model Architecture - Custom CNN
```
Conv2D(32) → MaxPool → 
Conv2D(64) → MaxPool → 
Conv2D(64) → MaxPool → 
Conv2D(64) → MaxPool → 
Conv2D(64) → MaxPool → 
Conv2D(64) → MaxPool → 
Flatten → Dense(64) → Dense(2)
```

## Slide 6: Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Sparse Categorical Crossentropy
- **Batch Size**: 16
- **Epochs**: 50 (with early stopping)
- **Callbacks**: 
  - Early Stopping (patience=10)
  - ReduceLROnPlateau
  - ModelCheckpoint

## Slide 7: Results
| Model | Test Accuracy |
|-------|--------------|
| **MobileNetV2** | **98.25%** ✓ |
| VGG16 | 78.95% |
| CustomCNN | 70.18% |
| ResNet50 | 42.11% |

**Best Model**: MobileNetV2

## Slide 8: CustomCNN Performance
```
                precision  recall  f1-score
A. OCP             0.67     0.87     0.75
C. Normal Eyes     0.78     0.52     0.62

Accuracy: 70.18%
```

## Slide 9: Web Application
**Features**:
- Upload eye images (JPG, PNG, TIF)
- Select model for prediction
- Real-time classification
- Confidence scores
- Ensemble predictions

**Tech Stack**: Streamlit / Flask

## Slide 10: Demo
[Live Demo Screenshot]
- Show uploaded image
- Model predictions
- Confidence scores

## Slide 11: Key Findings
✓ MobileNetV2 achieved 98.25% accuracy
✓ Transfer learning outperformed custom CNN
✓ Small dataset handled with augmentation
✓ Lightweight model suitable for deployment

## Slide 12: Challenges & Solutions
**Challenges**:
- Small dataset (285 images)
- Class imbalance
- Model compatibility issues

**Solutions**:
- Data augmentation
- Transfer learning with ImageNet weights
- Early stopping to prevent overfitting

## Slide 13: Future Work
- Collect more training data
- Multi-class classification (more eye conditions)
- Mobile app deployment
- Real-time video classification
- Explainable AI (Grad-CAM visualization)

## Slide 14: Conclusion
- Successfully built automated eye disease classifier
- MobileNetV2 achieved 98.25% accuracy
- Deployed as web application
- Ready for clinical validation

## Slide 15: Thank You
**Questions?**

Contact: [Your Email]
GitHub: [Repository Link]
