import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import VGG16

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
                image = cv2.resize(image, (64, 64))  # Resize to 64x64
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB
                images.append(image)
                labels.append(label)
            else:
                print(f"Warning: Unable to load image {image_path}")
    images = np.array(images).reshape(-1, 64, 64, 3) / 255.0  # Normalize
    return images, labels

# Load training and validation data
train_images, train_labels = load_data('hiragana/train/label.txt', 'hiragana/train')
valid_images, valid_labels = load_data('hiragana/valid/label.txt', 'hiragana/valid')
print(train_labels, valid_labels)

# Combine training and validation labels for fitting the label encoder
all_labels = list(set(train_labels + valid_labels))  # Use a set to ensure unique labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

train_labels = to_categorical(label_encoder.transform(train_labels), num_classes=len(label_encoder.classes_))
valid_labels = to_categorical(label_encoder.transform(valid_labels), num_classes=len(label_encoder.classes_))

print(label_encoder.classes_, len(label_encoder.classes_))

# Dynamically adjust the model's output layer to match the number of classes
num_classes = len(label_encoder.classes_)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))  # Changed to 3 channels
base_model.trainable = False  # Freeze base model layers

# Building the custom model on top of VGG16
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Added pooling to reduce feature map size
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax', name='output_layer')  # Adjust output layer dynamically
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
csv_logger = CSVLogger('training_log.csv', append=True, separator=';')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
checkpoint = ModelCheckpoint('hiragana_model.keras', monitor='val_loss', save_best_only=True, mode='min')

# Learning Rate Schedule function
def lr_schedule(epoch, lr):
    if epoch < 10:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))  # Exponential decay after 10 epochs

lr_scheduler = LearningRateScheduler(lr_schedule)

# Data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True  # Added flip for more variation
)

# Fine-tuning process: First train only the custom layers
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    epochs=50, validation_data=(valid_images, valid_labels),
                    callbacks=[csv_logger, reduce_lr, checkpoint, lr_scheduler])

# After 10 epochs, unfreeze the base model and fine-tune it
base_model.trainable = True  # Unfreeze the base model layers

# Recompile the model to apply the changes
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training with the entire model
history_finetune = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                             epochs=40, validation_data=(valid_images, valid_labels),
                             callbacks=[csv_logger, reduce_lr, checkpoint, lr_scheduler])

# Save the fine-tuned model
model.save('hiragana_finetuned_model.keras')

# Evaluate model
loss, accuracy = model.evaluate(valid_images, valid_labels)
print(f'Validation Accuracy after fine-tuning: {accuracy * 100:.2f}%')
