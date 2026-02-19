
# Crop & Grain Quality AI System

This project uses Machine Learning (TensorFlow/MobileNetV2) to detect:
1.  **Crop Diseases** (e.g., Potato Blight, Tomato Spot)
2.  **Grain Quality** (e.g., Damaged vs Healthy Wheat, Broken vs Full Rice)

## How to Run on Another Laptop (Setup Guide)

Follow these steps to set up the project on a new machine:

### 1. Install Python & Git
Make sure you have Python (3.9+) and Git installed.

### 2. Clone the Repository
Open a terminal and run:
```bash
git clone <YOUR_REPOSITORY_URL>
cd AGRXSETU
```

### 3. Install Dependencies
Install the required libraries:
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
The dataset is too large for Git, so run this script to download it automatically:
```bash
python download_grains.py
```
*This will create a `dataset` folder with Wheat and Rice images.*

### 5. Train the Model
Train the AI on the downloaded images:
```bash
python train.py
```
*Wait for the training to finish. It will save `crop_disease_model.h5`.*

### 6. Run the Camera (Test)
Start the webcam to test the model:
```bash
python inference.py
```
*Press 'q' to quit.*
