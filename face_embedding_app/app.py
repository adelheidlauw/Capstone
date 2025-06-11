from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load model embedding
model = tf.keras.models.load_model("face_embedding_model.h5")

# Preprocess image: resize to 160x160 and normalize
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((160, 160))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)  # shape: (1, 160, 160, 3)

@app.route('/')
def home():
    return "Face Embedding API is running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image_bytes = file.read()
    img_array = preprocess_image(image_bytes)

    embedding = model.predict(img_array)[0]  # shape: (512,)
    return jsonify({'embedding': embedding.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)