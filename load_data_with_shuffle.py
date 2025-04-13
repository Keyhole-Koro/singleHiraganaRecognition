import numpy as np
import os
import cv2

def load_data(label_file, image_dir, shuffle=True):
    images = []
    labels = []
    with open(label_file, 'r', encoding='utf-8') as file:  # Specify encoding
        for line in file:
            image_name, label = line.strip().split()
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:  # Check if image is loaded
                image = cv2.resize(image, (64, 64))  # Resize to 64x64
                images.append(image)
                labels.append(label)
            else:
                print(f"Warning: Unable to load image {image_path}")
    images = np.array(images).reshape(-1, 64, 64, 1) / 255.0  # Normalize
    labels = np.array(labels)

    if shuffle:
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images = images[indices]
        labels = labels[indices]

    return images, labels
