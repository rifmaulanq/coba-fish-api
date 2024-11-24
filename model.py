import tensorflow as tf
import cv2
import numpy as np

def load_model():
    """Loads the saved model."""
    return tf.keras.models.load_model("./model/fish_freshness_model.h5")

def preprocess_image(image_path, target_size):
    """
    Preprocess the input image for the model using OpenCV.
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size for resizing the image (width, height).

    Returns:
        numpy.ndarray: Preprocessed image ready for prediction.
    """
    image = cv2.imread(image_path)  # Read the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, target_size)  # Resize to the model's input size
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_image(model, image_path, target_size):
    """
    Predicts the label/text for the given image.
    Args:
        model (tensorflow.keras.Model): Loaded model.
        image_path (str): Path to the input image.
        target_size (tuple): Target size for resizing the image (width, height).

    Returns:
        str: Predicted label/text.
    """
    preprocessed_image = preprocess_image(image_path, target_size)
    prediction = model.predict(preprocessed_image)[0][0]  # Sigmoid output
    if prediction < 0.5:
        return "Fresh " + str(prediction)
    else:
        return "Not Fresh " + str(prediction)
