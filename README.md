# Ophthalmology Image Classification

Binary classification of eye images: **OCP vs Normal Eyes**

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python train_models.py
```

This will train 4 models:
- Custom CNN
- VGG16 (Transfer Learning)
- ResNet50 (Transfer Learning)
- MobileNetV2 (Transfer Learning)

Models saved in `models/` directory.

### 3. Run Streamlit App
```bash
streamlit run app.py
```

## 📁 Project Structure
```
├── Dataset/
│   ├── A. OCP/
│   └── C. Normal Eyes/
├── models/              # Trained models (created after training)
├── results/             # Confusion matrices (created after training)
├── train_models.py      # Training script
├── app.py               # Streamlit frontend
├── utils.py             # Helper functions
└── requirements.txt     # Dependencies
```

## 🎯 Features
- Data augmentation for better generalization
- Transfer learning with ImageNet weights
- Early stopping & learning rate scheduling
- Ensemble predictions in Streamlit app
- Real-time confidence scores

## 📊 Expected Accuracy
- Transfer Learning Models: 92-97%
- Custom CNN: 85-90%
