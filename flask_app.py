"""
OphthoAI -- Professional Flask Application
OCP vs Normal Eyes Classifier with Grad-CAM Explainability
Memory-optimized for cloud deployment (Render free tier)
"""

import os
import gc
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import base64

# ---- App Setup ----
app = Flask(__name__)
IMG_SIZE = (256, 256)

# ---- Lazy Model Loading (memory-efficient) ----
# Only load one model at a time to stay within 512MB RAM
class_names = np.load('models/class_names.npy', allow_pickle=True)

AVAILABLE_MODELS = []
for name in ['MobileNetV2', 'CustomCNN', 'VGG16', 'ResNet50']:
    if os.path.exists(f'models/{name}.keras'):
        AVAILABLE_MODELS.append(name)

print(f'[INIT] Available models: {AVAILABLE_MODELS}')

_loaded_model = None
_loaded_model_name = None


def get_model(name):
    """Load a model, unloading the previous one to save memory."""
    global _loaded_model, _loaded_model_name
    import tensorflow as tf

    if _loaded_model_name == name and _loaded_model is not None:
        return _loaded_model

    # Unload previous model
    if _loaded_model is not None:
        del _loaded_model
        _loaded_model = None
        _loaded_model_name = None
        gc.collect()
        try:
            from tensorflow.keras import backend as K
            K.clear_session()
        except Exception:
            pass

    print(f'[LOAD] Loading {name}...', flush=True)
    _loaded_model = tf.keras.models.load_model(f'models/{name}.keras')

    # Warmup
    try:
        dummy = tf.zeros((1, *IMG_SIZE, 3), dtype=tf.float32)
        _ = _loaded_model(dummy, training=False)
    except Exception:
        pass

    _loaded_model_name = name
    print(f'[LOAD] {name} ready', flush=True)
    return _loaded_model


# ---- Grad-CAM Utilities ----
def _get_last_conv_layer(model):
    """Walk layers recursively to find final Conv2D (handles nested models)."""
    import tensorflow as tf

    def collect(obj):
        out = []
        if not hasattr(obj, 'layers'):
            return out
        for l in obj.layers:
            out.append(l)
            out.extend(collect(l))
        return out

    for layer in reversed(collect(model)):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, class_index=None):
    import tensorflow as tf
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    try:
        _ = model(img_tensor, training=False)
    except Exception:
        return None

    last_conv_name = _get_last_conv_layer(model)
    if last_conv_name is None:
        return None

    conv_layer = model.get_layer(last_conv_name)
    grad_model = tf.keras.models.Model(
        [model.inputs], [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return None

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()


def overlay_heatmap_on_image(heatmap, image, alpha=0.4):
    import matplotlib.cm as cm
    if heatmap is None:
        return None
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap_uint8).resize(image.size)
    heatmap_arr = np.array(heatmap_img) / 255.0
    colored = cm.get_cmap('jet')(heatmap_arr)[..., :3]
    colored = np.uint8(255 * colored)
    overlay = Image.fromarray(colored).convert('RGBA')
    base = image.convert('RGBA')
    return Image.blend(base, overlay, alpha=alpha)


def _image_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ---- Helper: Decode + Preprocess ----
def decode_image(b64_string):
    raw = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    img_resized = img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img_resized).astype(np.float32) / 255.0, axis=0)
    return img, img_array


def _run_prediction(model_name, img, img_array):
    """Run single model prediction with Grad-CAM."""
    model = get_model(model_name)
    pred = model.predict(img_array, verbose=0)
    cls_idx = int(np.argmax(pred))

    heatmap_b64 = None
    try:
        heatmap = make_gradcam_heatmap(img_array, model, class_index=cls_idx)
        if heatmap is not None:
            cam = overlay_heatmap_on_image(heatmap, img)
            if cam is not None:
                heatmap_b64 = _image_to_base64(cam)
    except Exception as e:
        print(f'[XAI-ERROR] {model_name}: {e}', flush=True)

    return {
        'model': model_name,
        'class': str(class_names[cls_idx]),
        'confidence': f'{pred[0][cls_idx] * 100:.1f}',
        'probs': list(class_names),
        'prob_values': [f'{p * 100:.1f}' for p in pred[0]],
        'heatmap': heatmap_b64,
    }


# ---- Routes ----
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/models_list')
def models_list():
    return jsonify({'models': AVAILABLE_MODELS})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_name = data.get('model', 'MobileNetV2')

    if model_name not in AVAILABLE_MODELS:
        return jsonify({'error': f'Model {model_name} not available'}), 400

    try:
        img, img_array = decode_image(data['image'])
        result = _run_prediction(model_name, img, img_array)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    try:
        img, img_array = decode_image(data['image'])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    results = {}
    all_preds = []
    for name in AVAILABLE_MODELS:
        result = _run_prediction(name, img, img_array)
        results[name] = result
        all_preds.append([float(v) for v in result['prob_values']])

    # Ensemble
    ensemble_probs = np.mean(all_preds, axis=0)
    ens_idx = int(np.argmax(ensemble_probs))
    ensemble = {
        'class': str(class_names[ens_idx]),
        'confidence': f'{ensemble_probs[ens_idx]:.1f}',
    }

    return jsonify({'models': results, 'ensemble': ensemble})


# ---- Main ----
if __name__ == '__main__':
    print(f'\n  OphthoAI -- {len(AVAILABLE_MODELS)} models available (lazy loading)')
    print(f'  Classes: {list(class_names)}')
    print(f'  http://localhost:5000\n')
    app.run(debug=True, port=5000)
