import streamlit as st
import numpy as np
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

st.set_page_config(page_title="Ophthalmology Classifier", page_icon="👁️", layout="wide")

@st.cache_resource
def load_models():
    import tensorflow as tf
    models = {}
    for name in ['CustomCNN', 'VGG16', 'ResNet50', 'MobileNetV2']:
        path = f'models/{name}'
        if os.path.exists(path):
            models[name] = tf.keras.models.load_model(path)
    class_names = np.load('models/class_names.npy', allow_pickle=True)
    return models, class_names

st.title("👁️ Ophthalmology Image Classifier")
st.markdown("**Binary Classification: OCP vs Normal Eyes**")

try:
    models, class_names = load_models()
    st.sidebar.success(f"✅ {len(models)} models loaded")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Model selection
st.sidebar.markdown("---")
st.sidebar.subheader("Select Model")
selected_models = st.sidebar.multiselect(
    "Choose models for prediction:",
    list(models.keys()),
    default=list(models.keys())
)

use_ensemble = st.sidebar.checkbox("Show Ensemble Prediction", value=True)

uploaded_file = st.file_uploader("Upload eye image", type=['jpg', 'jpeg', 'png', 'tif'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image.resize((256, 256))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Uploaded Image", width=300)
    
    with col2:
        st.subheader("🔍 Predictions")
        
        if not selected_models:
            st.warning("Please select at least one model")
        else:
            all_preds = []
            for name in selected_models:
                pred = models[name].predict(img_array, verbose=0)
                cls = class_names[np.argmax(pred)]
                conf = np.max(pred) * 100
                all_preds.append(pred)
                
                st.write(f"**{name}**: {cls} ({conf:.1f}%)")
                st.progress(int(conf))
            
            # Ensemble
            if use_ensemble and len(selected_models) > 1:
                st.markdown("---")
                ensemble = np.mean(all_preds, axis=0)
                final_cls = class_names[np.argmax(ensemble)]
                final_conf = np.max(ensemble) * 100
                
                st.success(f"**Ensemble: {final_cls}** ({final_conf:.1f}%)")
                st.progress(int(final_conf))
