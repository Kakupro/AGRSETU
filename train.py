
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import datetime

# --- Configuration ---
DATA_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = "crop_disease_model.h5"
LABELS_SAVE_PATH = "labels.txt"

def train_model():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found!")
        print("Please create a 'dataset' folder and put your class folders inside it.")
        print("Example structure:")
        print("  dataset/")
        print("    healthy_wheat/")
        print("    diseased_wheat/")
        print("    good_quality_rice/")
        print("    broken_rice/")
        return

    print("Loading data...")
    
    # 1. Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    if train_generator.samples == 0:
        print("No images found! Make sure you have images in subfolders inside 'dataset'.")
        return

    class_names = list(train_generator.class_indices.keys())
    print(f"Classes found: {class_names}")
    
    # Save labels for inference
    with open(LABELS_SAVE_PATH, 'w') as f:
        for cls in class_names:
            f.write(f"{cls}\n")

    # 2. Build Model (Transfer Learning with MobileNetV2)
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False # Freeze base layers initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 3. Train
    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # 4. Save Model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Labels saved to {LABELS_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
