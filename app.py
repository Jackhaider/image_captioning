# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/caption', methods=['POST'])
def generate_caption():
    image_file = request.files['image']
    image = Image.open(image_file).resize((224, 224))
    # Preprocess image and use model to predict caption
    caption = model.predict(image)  # Adjust according to your input requirements
    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)
