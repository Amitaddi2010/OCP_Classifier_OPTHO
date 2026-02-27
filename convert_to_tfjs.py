import tensorflowjs as tfjs
import tensorflow as tf

models = ['CustomCNN', 'VGG16', 'ResNet50', 'MobileNetV2']

for name in models:
    try:
        model = tf.keras.models.load_model(f'models/{name}.keras')
        tfjs.converters.save_keras_model(model, f'models/{name}')
        print(f"Converted {name}")
    except Exception as e:
        print(f"Error converting {name}: {e}")
