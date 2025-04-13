import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from load_data_with_shuffle import load_data

# Ensure GPU is used if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_images, train_labels = load_data('hiragana44/train/label.txt', 'hiragana44/train')
valid_images, valid_labels = load_data('hiragana44/valid/label.txt', 'hiragana44/valid')

label_encoder = LabelEncoder()
train_labels = to_categorical(label_encoder.fit_transform(train_labels))
valid_labels = to_categorical(label_encoder.transform(valid_labels))

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    Flatten(),  # Remove one MaxPooling2D layer to avoid negative dimensions
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer=Adadelta(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

csv_logger = CSVLogger('training_log.csv', append=True, separator=';')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
checkpoint = ModelCheckpoint('hiragana_vgg_best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

def lr_schedule(epoch, lr):
    if epoch < 10:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))

lr_scheduler = LearningRateScheduler(lr_schedule)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    epochs=1000, validation_data=(valid_images, valid_labels),
                    callbacks=[csv_logger, reduce_lr, checkpoint, lr_scheduler])

model.save('hiragana_vgg_model.keras')  # Save the final trained model in Keras format

loss, accuracy = model.evaluate(valid_images, valid_labels)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
