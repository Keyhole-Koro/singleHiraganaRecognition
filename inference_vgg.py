import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

def load_image(image_path, image_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        image = cv2.resize(image, image_size)
        image = image.reshape(1, image_size[0], image_size[1], 1) / 255.0  # Normalize
    else:
        raise ValueError(f"Unable to load image {image_path}")
    return image

def load_label_encoder(label_file):
    labels = []
    with open(label_file, 'r', encoding='utf-8') as file:
        for line in file:
            _, label = line.strip().split()
            labels.append(label)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    return label_encoder

# Load the fine-tuned VGG model
model = load_model('hiragana_vgg_model_finetuned.h5')

# Load the label encoder
label_encoder = load_label_encoder('hiragana44/train/label.txt')

# Predict function
def predict(image_path):
    image = load_image(image_path, (32, 32))
    predictions = model.predict(image)
    predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])
    return predicted_label[0]

# Test an image and print the prediction
def test_image(image_path):
    prediction = predict(image_path)
    print(f'Predicted label for {image_path}: {prediction}')

# Example usage
if __name__ == "__main__":
    test_image_path = 'hiragana44/valid/U304A_0.png'  # Replace with the path to your test image
    test_image(test_image_path)
