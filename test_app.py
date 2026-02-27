import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Ophthalmology Classifier", page_icon="👁️", layout="wide")

@st.cache_resource
def load_models():
    from tensorflow import keras
    models = {}
    for name in ['CustomCNN', 'VGG16', 'ResNet50', 'MobileNetV2']:
        path = f'models/{name}.h5'
        if os.path.exists(path):
            models[name] = keras.models.load_model(path)
    class_names = np.load('models/class_names.npy', allow_pickle=True)
    return models, class_names

st.title("👁️ Ophthalmology Image Classifier")
st.markdown("**Binary Classification: OCP vs Normal Eyes**")

try:
    models, class_names = load_models()
    st.sidebar.success(f"✅ {len(models)} models loaded")
except:
    st.error("Models not found. Run: python train_models.py")
    st.stop()

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
        
        for name, model in models.items():
            pred = model.predict(img_array, verbose=0)
            cls = class_names[np.argmax(pred)]
            conf = np.max(pred) * 100
            
            st.write(f"**{name}**: {cls} ({conf:.1f}%)")
            st.progress(int(conf))
        
        # Ensemble
        st.markdown("---")
        all_preds = [model.predict(img_array, verbose=0) for model in models.values()]
        ensemble = np.mean(all_preds, axis=0)
        final_cls = class_names[np.argmax(ensemble)]
        final_conf = np.max(ensemble) * 100
        
        st.success(f"**Ensemble: {final_cls}** ({final_conf:.1f}%)")
        st.progress(int(final_conf))
