import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Ensure GPU is used if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_data(label_file, image_dir):
    images = []
    labels = []
    with open(label_file, 'r', encoding='utf-8') as file:  # Specify encoding
        for line in file:
            image_name, label = line.strip().split()
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:  # Check if image is loaded
                image = cv2.resize(image, (32, 32))  # Resize to 32x32
                images.append(image)
                labels.append(label)
            else:
                print(f"Warning: Unable to load image {image_path}")
    images = np.array(images).reshape(-1, 32, 32, 1) / 255.0  # Normalize
    return images, labels

train_images, train_labels = load_data('hiragana44/train/label.txt', 'hiragana44/train')
valid_images, valid_labels = load_data('hiragana44/valid/label.txt', 'hiragana44/valid')

label_encoder = LabelEncoder()
train_labels = to_categorical(label_encoder.fit_transform(train_labels))
valid_labels = to_categorical(label_encoder.transform(valid_labels))

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    Flatten(),  # Remove one MaxPooling2D layer to avoid negative dimensions
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])

csv_logger = CSVLogger('training_log.csv', append=True, separator=';')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
checkpoint = ModelCheckpoint('hiragana_vgg_best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    epochs=1000, validation_data=(valid_images, valid_labels),
                    callbacks=[csv_logger, early_stopping, reduce_lr, checkpoint])

model.save('hiragana_vgg_model.h5')  # Save the final trained model

loss, accuracy = model.evaluate(valid_images, valid_labels)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
