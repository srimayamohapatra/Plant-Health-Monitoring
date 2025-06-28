## Plant Disease Detection

Overview

This project uses a convolutional neural network (CNN) implemented in TensorFlow/Keras to detect plant diseases from images.

Objective

Classify images of plant leaves into different disease categories.

Dataset

The dataset contains multiple plant species and disease classes, including:

- Apple: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- Blueberry: Healthy
- Cherry: Healthy, Powdery Mildew
- Corn (Maize): Cercospora Leaf Spot Gray Leaf Spot, Common Rust, Healthy, Northern Leaf Blight

Model Architecture


model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(256,256,3)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(11, activation='softmax')  # 11 output classes
])


Training

- Training/Validation Split: 80% training, 20% validation (from training set)
- Batch Size: 10
- Epochs: 50
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy

Training Results

- The model achieves near-perfect training accuracy and a validation accuracy around 62%.
- Test accuracy is approximately 56.8%.

Usage

1. Prepare Dataset: Place images in folders named by disease type inside a train directory.
2. Run Script: Execute the provided Python script to preprocess images, train the model, and evaluate performance.
3. Save Model: The trained model is saved as plant_disease_model.h5 (or .keras for newer Keras versions).
4. Predict: Use the model to predict disease classes on new images.

Example Output

| Sample | True Label | Predicted Label |
| --- | --- | --- |
| 1 | Corn_(maize)_healthy | Corn(maize)_healthy |
| 2 | Corn(maize)_Common_rust | Corn(maize)_Common_rust |
| 3 | Apple__healthy | Apple__healthy |
| 4 | Corn(maize)__Northern_Leaf_Blight | Corn(maize)_Common_rust |
| 5 | Blueberry__healthy | Cherry(including_sour)__healthy |
