import os
import numpy as np
import tensorflow as tf
import argparse

# Load the trained model
model_path = 'metal_defect_model.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at path: {model_path}")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at path: {model_path}")

model = tf.keras.models.load_model(model_path)

# Function to test the model on a sample image
def predict_image(model, image_path, class_indices):
    # Load the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    input_data = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img) / 255.0, axis=0)

    # Make prediction
    predictions = model.predict(input_data)

    # Map the predicted label indices to the class names
    class_labels = {v: k for k, v in class_indices.items()}

    # Print all predictions with their probabilities
    for idx, probability in enumerate(predictions[0]):
        if probability >= 0.1:
            label_name = class_labels[idx]
            print(f'{label_name}: {probability * 100:.2f}%')

# Load class indices from JSON file
import json

class_indices_path = 'class_indices.json'
if not os.path.exists(class_indices_path):
    raise FileNotFoundError(f"Class indices file not found at path: {class_indices_path}")

with open(class_indices_path, 'r') as json_file:
    class_indices = json.load(json_file)

# Command line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the trained model on an input image.')
    parser.add_argument('image_path', type=str, help='Path to the image file for evaluation.')
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found at path: {args.image_path}")

    # Predict the class of the input image
    predict_image(model, args.image_path, class_indices)
