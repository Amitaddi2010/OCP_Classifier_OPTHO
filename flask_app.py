from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.cm as cm

app = Flask(__name__)

IMG_SIZE = (256, 256)

models = {}
class_names = np.load('models/class_names.npy', allow_pickle=True)

for name in ['CustomCNN', 'VGG16', 'ResNet50', 'MobileNetV2']:
    try:
        models[name] = tf.keras.models.load_model(f'models/{name}.keras')
    except Exception:
        pass


def _warmup_model(model):
    try:
        dummy = tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)
        _ = model(dummy, training=False)
        return True
    except Exception:
        return False


for _m in models.values():
    _warmup_model(_m)


def _get_last_conv_layer(model):
    # Walk layers recursively so we can find conv layers inside nested models
    def collect_layers(obj):
        collected = []
        if not hasattr(obj, "layers"):
            return collected
        for layer in obj.layers:
            collected.append(layer)
            collected.extend(collect_layers(layer))
        return collected

    all_layers = collect_layers(model)
    for layer in reversed(all_layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, class_index=None):
    # Ensure model is built/called so model.inputs/model.output exist (fixes Sequential "never been called" error)
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
        [model.inputs],
        [conv_layer.output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
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


HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Eye Classifier</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        #preview { max-width: 400px; margin: 20px 0; }
        .result { margin: 20px 0; padding: 15px; background: #f0f0f0; border-radius: 5px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        select { padding: 8px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>👁️ Eye Image Classifier</h1>
    <p>OCP vs Normal Eyes</p>
    
    <select id="model">
        <option value="MobileNetV2">MobileNetV2</option>
        <option value="VGG16">VGG16</option>
        <option value="CustomCNN">CustomCNN</option>
        <option value="ResNet50">ResNet50</option>
    </select>
    
    <br><br>
    <input type="file" id="upload" accept="image/*">
    <br><br>
    <img id="preview" style="display:none;">
    <br>
    <button onclick="predict()" id="predictBtn" style="display:none;">Predict</button>
    
    <div id="result" class="result" style="display:none;"></div>
    <img id="heatmap" style="max-width:400px; margin-top:20px; display:none;">

    <script>
        let imageData;
        
        document.getElementById('upload').onchange = function(e) {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = function(event) {
                document.getElementById('preview').src = event.target.result;
                document.getElementById('preview').style.display = 'block';
                document.getElementById('predictBtn').style.display = 'inline-block';
                imageData = event.target.result.split(',')[1];
            };
            reader.readAsDataURL(file);
        };
        
        async function predict() {
            const model = document.getElementById('model').value;
            document.getElementById('result').innerHTML = 'Predicting...';
            document.getElementById('result').style.display = 'block';
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model: model, image: imageData})
            });
            
            const data = await response.json();
            document.getElementById('result').innerHTML = `
                <h3>Prediction: ${data.class}</h3>
                <p>Confidence: ${data.confidence}%</p>
                <p>${data.probs[0]}: ${data.prob_values[0]}%</p>
                <p>${data.probs[1]}: ${data.prob_values[1]}%</p>
            `;

            if (data.heatmap) {
                const heatmapImg = document.getElementById('heatmap');
                heatmapImg.src = 'data:image/png;base64,' + data.heatmap;
                heatmapImg.style.display = 'block';
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_name = data['model']
    img_data = base64.b64decode(data['image'])
    
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = (np.array(img).astype(np.float32) / 255.0)
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = models[model_name].predict(img_array, verbose=0)
    cls_idx = np.argmax(pred)

    # Try to generate Grad-CAM, but never fail the request if it breaks
    heatmap_b64 = None
    try:
        heatmap = make_gradcam_heatmap(img_array, models[model_name], class_index=cls_idx)
        if heatmap is not None:
            cam_image = overlay_heatmap_on_image(heatmap, img)
            if cam_image is not None:
                buffer = io.BytesIO()
                cam_image.save(buffer, format="PNG")
                heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        print(
            f"[XAI] model={model_name}, class={class_names[cls_idx]}, "
            f"heatmap_generated={heatmap is not None if 'heatmap' in locals() else False}, "
            f"heatmap_b64_len={len(heatmap_b64) if heatmap_b64 else 0}",
            flush=True,
        )
    except Exception as e:
        print(f"[XAI-ERROR] model={model_name}, error={e}", flush=True)
        heatmap_b64 = None
    
    return jsonify({
        'class': class_names[cls_idx],
        'confidence': f'{pred[0][cls_idx] * 100:.1f}',
        'probs': list(class_names),
        'prob_values': [f'{p * 100:.1f}' for p in pred[0]],
        'heatmap': heatmap_b64,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
