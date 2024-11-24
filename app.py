from flask import Flask, request, jsonify
import os
from model import load_model, predict_image  # Import from model.py
import cv2
import numpy as np

app = Flask(__name__)

# Load the model once at the start
model = load_model()

@app.route('/ikan', methods=['POST'])
def identifikasi():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Retrieve the image from the request
    image_file = request.files['image']

    # Save the image temporarily
    temp_path = os.path.join('temp_image.jpg')
    image_file.save(temp_path)

    try:
        # Call the prediction    function
        prediction = predict_image(model, temp_path, target_size=(128, 128))
        os.remove(temp_path)  # Remove temporary file
        return jsonify({'prediction': prediction})
    except Exception as e:
        os.remove(temp_path)  # Ensure file is removed even on error
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
