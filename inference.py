
import cv2
import numpy as np
import tensorflow as tf
import os

# --- Configuration ---
MODEL_PATH = "crop_disease_model.h5"
LABELS_PATH = "labels.txt"
CAMERA_ID = 0  # Camera index (usually 0 for default webcam)
IMG_SIZE = (224, 224)

def load_labels(labels_file):
    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def infer_disease():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found! Please run 'python train.py' and ensure you have data.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
    print(labels)

    print("Starting webcam...")
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("Cannot open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display Frame
        original_frame = frame.copy()

        # Resize for model input
        frame_resized = cv2.resize(frame, IMG_SIZE)
        
        # Preprocess for MobileNetV2 inputs: scale to [0,1]
        frame_input = np.expand_dims(frame_resized, axis=0)
        frame_input = frame_input.astype('float32') / 255.0

        # Predict
        predictions = model.predict(frame_input)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        label = labels[class_idx]

        # Draw Prediction on Frame
        cv2.putText(original_frame, f"{label}: {confidence:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show Results
        cv2.imshow('Crop Disease & Grain Quality Detector', original_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    infer_disease()
