Plant Disease Detection
This project is designed to detect plant diseases from images using a convolutional neural network (CNN) implemented in TensorFlow/Keras.

Overview
Objective: Classify images of plant leaves into different disease categories.

Dataset: Images are organized into folders by disease type (e.g., Apple___Apple_scab, Corn_(maize)___healthy).

Model: A custom CNN architecture is used for classification.

Output: Model predicts the type of plant disease or healthy status from input images.

Dataset
The dataset contains multiple plant species and disease classes, including:

Apple: Apple Scab, Black Rot, Cedar Apple Rust, Healthy

Blueberry: Healthy

Cherry: Healthy, Powdery Mildew

Corn (Maize): Cercospora Leaf Spot Gray Leaf Spot, Common Rust, Healthy, Northern Leaf Blight

Images are resized to 256x256 pixels and normalized.

Model Architecture
python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(256,256,3)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(11, activation='softmax')  # 11 output classes
])
Note: The architecture in your code may differ; adjust this section to match your actual model if needed.

Training
Training/Validation Split: 80% training, 20% validation (from training set).

Batch Size: 10

Epochs: 50

Optimizer: Adam

Loss: Categorical Crossentropy

Metrics: Accuracy

Training Results:
The model achieves near-perfect training accuracy and a validation accuracy around 62%. Test accuracy is approximately 56.8%.

Usage
Prepare Dataset:
Place images in folders named by disease type inside a train directory.

Run Script:
Execute the provided Python script to preprocess images, train the model, and evaluate performance.

Save Model:
The trained model is saved as plant_disease_model.h5 (or .keras for newer Keras versions).

Predict:
Use the model to predict disease classes on new images.

Example Output
text
Sample 1: True = Corn_(maize)___healthy | Predicted = Corn_(maize)___healthy
Sample 2: True = Corn_(maize)___Common_rust_ | Predicted = Corn_(maize)___Common_rust_
Sample 3: True = Apple___healthy | Predicted = Apple___healthy
Sample 4: True = Corn_(maize)___Northern_Leaf_Blight | Predicted = Corn_(maize)___Common_rust_
Sample 5: True = Blueberry___healthy | Predicted = Cherry_(including_sour)___healthy
...
