import os
import numpy as np
import requests
from PIL import Image
from io import BytesIO

from flask import Flask, request, jsonify
import tensorflow as tf

# === Konfigurasi model ===
MODEL_URL = "https://drive.google.com/uc?export=download&id=1sixXpc0Mesa2TcNkeBInFgsre5QEPKwv"
MODEL_PATH = "face_embedding_model.h5"

# === Download model jika belum ada ===
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded.")

# === Load model ===
embedding_model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded.")

# === Inisialisasi Flask app ===
app = Flask(__name__)

# === Fungsi preprocessing gambar ===
def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Tambah batch dimensi
    return img_array

# === Endpoint utama untuk prediksi embedding ===
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["file"]
    image_bytes = file.read()
    
    try:
        preprocessed_img = preprocess_image(image_bytes)
        embedding = embedding_model.predict(preprocessed_img)[0]  # 512-d vector
        return jsonify({
            "embedding": embedding.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Root endpoint ===
@app.route("/", methods=["GET"])
def index():
    return "Face Embedding API is running."

# === Jalankan app ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))