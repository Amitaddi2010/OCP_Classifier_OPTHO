import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import matplotlib.cm as cm

# Page config
st.set_page_config(page_title="Ophthalmology Classifier", layout="wide", page_icon="👁️")

# Load models
@st.cache_resource
def load_models():
    from tensorflow import keras
    models = {}
    model_names = ['CustomCNN', 'VGG16', 'ResNet50', 'MobileNetV2']
    
    for name in model_names:
        model_path = f'models/{name}.h5'
        if os.path.exists(model_path):
            try:
                models[name] = keras.saving.load_model(model_path, compile=False)
            except Exception:
                models[name] = keras.models.load_model(model_path, compile=False, safe_mode=False)
    
    class_names = np.load('models/class_names.npy', allow_pickle=True)
    return models, class_names

def preprocess_image(image):
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def _get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, class_index=None):
    last_conv_name = _get_last_conv_layer(model)
    if last_conv_name is None:
        return None
    conv_layer = model.get_layer(last_conv_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [conv_layer.output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return None
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def overlay_heatmap_on_image(heatmap, image, alpha=0.4):
    if heatmap is None:
        return None
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap).resize(image.size)
    heatmap_arr = np.array(heatmap_img) / 255.0
    colored = cm.get_cmap("jet")(heatmap_arr)[..., :3]
    colored = np.uint8(255 * colored)
    overlay = Image.fromarray(colored).convert("RGBA")
    base = image.convert("RGBA")
    return Image.blend(base, overlay, alpha=alpha)

def predict(models, img_array, class_names):
    predictions = {}
    for name, model in models.items():
        pred = model.predict(img_array, verbose=0)
        predictions[name] = {
            'class': class_names[np.argmax(pred)],
            'confidence': float(np.max(pred) * 100),
            'probabilities': {class_names[i]: float(pred[0][i] * 100) for i in range(len(class_names))},
            'raw': pred,
        }
    return predictions

# Main app
st.title("👁️ Ophthalmology Image Classifier")
st.markdown("Upload an eye image to classify it as **OCP** or **Normal**")

# Load models
try:
    models, class_names = load_models()
    st.sidebar.success(f"✅ {len(models)} models loaded")
    st.sidebar.write("**Available Models:**")
    for name in models.keys():
        st.sidebar.write(f"- {name}")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an eye image...", type=['jpg', 'jpeg', 'png', 'tif'])

if uploaded_file:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("Analyzing image..."):
            img_array = preprocess_image(image)
            predictions = predict(models, img_array, class_names)
        
        st.subheader("🔍 Predictions with Explanations")
        
        for model_name, result in predictions.items():
            with st.expander(f"**{model_name}** - {result['class']} ({result['confidence']:.2f}%)", expanded=True):
                st.progress(result['confidence'] / 100)
                
                cols = st.columns(len(class_names))
                for idx, (cls, prob) in enumerate(result['probabilities'].items()):
                    with cols[idx]:
                        st.metric(cls, f"{prob:.2f}%")

                try:
                    heatmap = make_gradcam_heatmap(img_array, models[model_name])
                    cam_image = overlay_heatmap_on_image(heatmap, image)
                    if cam_image is not None:
                        st.image(cam_image, caption="Grad-CAM explanation", use_container_width=True)
                except Exception as e:
                    st.info(f"Explanation not available for {model_name}: {e}")
        
        # Ensemble prediction
        st.subheader("🎯 Ensemble Prediction")
        ensemble_probs = {}
        for cls in class_names:
            ensemble_probs[cls] = np.mean([pred['probabilities'][cls] for pred in predictions.values()])
        
        final_class = max(ensemble_probs, key=ensemble_probs.get)
        final_conf = ensemble_probs[final_class]
        
        st.success(f"**Final Prediction: {final_class}** (Confidence: {final_conf:.2f}%)")
        
        cols = st.columns(len(class_names))
        for idx, (cls, prob) in enumerate(ensemble_probs.items()):
            with cols[idx]:
                st.metric(f"Ensemble {cls}", f"{prob:.2f}%")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info(f"**Classes:** {', '.join(class_names)}\n\n**Image Size:** 256x256")
