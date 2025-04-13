import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import VGG16

# Ensure GPU is used if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_data(label_file, image_dir, image_size, channels):
    images = []
    labels = []
    with open(label_file, 'r', encoding='utf-8') as file:
        for line in file:
            image_name, label = line.strip().split()
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.resize(image, image_size)
                if channels == 1:
                    image = image.reshape(image_size[0], image_size[1], 1)
                images.append(image)
                labels.append(label)
            else:
                print(f"Warning: Unable to load image {image_path}")
    images = np.array(images) / 255.0  # Normalize
    return images, labels

def build_vgg16_model(num_classes, input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', name='output_layer')
    ])
    return model

def build_custom_model(num_classes, input_shape):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
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
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_and_finetune(model, train_images, train_labels, valid_images, valid_labels, output_model_path, epochs, batch_size):
    # Explicitly compile the model before training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    csv_logger = CSVLogger('training_log.csv', append=True, separator=';')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    checkpoint = ModelCheckpoint(output_model_path, monitor='val_loss', save_best_only=True, mode='min')

    def lr_schedule(epoch, lr):
        if epoch < 10:
            return float(lr)
        else:
            return float(lr * tf.math.exp(-0.1))

    lr_scheduler = LearningRateScheduler(lr_schedule)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Train the model
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
                        epochs=epochs, validation_data=(valid_images, valid_labels),
                        callbacks=[csv_logger, reduce_lr, checkpoint, lr_scheduler])

    # Fine-tune the model if it is VGG16-based
    if isinstance(model.layers[0], tf.keras.layers.Layer) and model.layers[0].name.startswith("vgg16"):
        model.layers[0].trainable = True  # Unfreeze the base model layers
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        history_finetune = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
                                     epochs=epochs, validation_data=(valid_images, valid_labels),
                                     callbacks=[csv_logger, reduce_lr, checkpoint, lr_scheduler])

    return model

if __name__ == "__main__":
    # Allow user to specify root folders for training and fine-tuning datasets
    train_root = "hiragana44"
    fine_tune_root = "hiragana"
    model_type = "custom"  # Change to "vgg16" for VGG16-based model

    # Derive paths for training and validation datasets
    train_label_file = os.path.join(train_root, "train", "label.txt")
    train_image_dir = os.path.join(train_root, "train")
    valid_label_file = os.path.join(train_root, "valid", "label.txt")
    valid_image_dir = os.path.join(train_root, "valid")

    # Derive paths for fine-tuning datasets
    fine_tune_label_file = os.path.join(fine_tune_root, "train", "label.txt")
    fine_tune_image_dir = os.path.join(fine_tune_root, "train")

    # Determine the number of channels based on the model type
    channels = 3 if model_type == "vgg16" else 1

    # Load training, validation, and fine-tuning data
    train_images, train_labels = load_data(train_label_file, train_image_dir, (64, 64), channels)
    valid_images, valid_labels = load_data(valid_label_file, valid_image_dir, (64, 64), channels)
    fine_tune_images, fine_tune_labels = load_data(fine_tune_label_file, fine_tune_image_dir, (64, 64), channels)

    # Combine all unique labels for fitting the label encoder
    all_labels = list(set(train_labels + valid_labels + fine_tune_labels))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    train_labels = to_categorical(label_encoder.transform(train_labels), num_classes=len(label_encoder.classes_))
    valid_labels = to_categorical(label_encoder.transform(valid_labels), num_classes=len(label_encoder.classes_))
    fine_tune_labels = to_categorical(label_encoder.transform(fine_tune_labels), num_classes=len(label_encoder.classes_))

    # Ensure the model's output layer matches the number of classes
    num_classes = len(label_encoder.classes_)
    if model_type == "vgg16":
        model = build_vgg16_model(num_classes, (64, 64, 3))
    else:
        model = build_custom_model(num_classes, (64, 64, 1))

    # Train the model
    output_model_path = 'hiragana_finetuned_model.keras' if model_type == "vgg16" else 'hiragana_custom_model.keras'
    model = train_and_finetune(model, train_images, train_labels, valid_images, valid_labels, output_model_path, epochs=30, batch_size=32)

    # Fine-tune the model using the fine-tuning dataset
    print("Starting fine-tuning...")
    model.layers[0].trainable = True  # Unfreeze the base model layers if VGG16
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    fine_tune_history = model.fit(fine_tune_images, fine_tune_labels, batch_size=32, epochs=50,
                                  validation_data=(valid_images, valid_labels),
                                  callbacks=[CSVLogger('fine_tuning_log.csv', append=True, separator=';'),
                                             ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5),
                                             ModelCheckpoint(output_model_path, monitor='val_loss', save_best_only=True, mode='min')])

    # Save the fine-tuned model
    model.save(output_model_path)

    # Evaluate the model
    loss, accuracy = model.evaluate(valid_images, valid_labels)
    print(f'Validation Accuracy after fine-tuning: {accuracy * 100:.2f}%')
