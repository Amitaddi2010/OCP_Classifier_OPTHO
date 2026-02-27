import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Ophthalmology Classifier", layout="wide", page_icon="👁️")

st.title("👁️ Ophthalmology Image Classifier")
st.markdown("Upload an eye image to classify it as **OCP** or **Normal**")

st.error("⚠️ TensorFlow installation is broken. Please run in terminal:")
st.code("pip uninstall -y tensorflow tensorflow-intel keras\npip install tensorflow==2.15.0", language="bash")

st.info("After fixing TensorFlow, run: `streamlit run app.py`")

uploaded_file = st.file_uploader("Choose an eye image...", type=['jpg', 'jpeg', 'png', 'tif'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", width=300)
    st.warning("Models cannot be loaded until TensorFlow is fixed.")
