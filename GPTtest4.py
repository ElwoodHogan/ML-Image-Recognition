import os
import numpy as np
from PIL import Image

# Constants
data_dir = "your_dataset_directory/"
image_size = (100, 100)
num_classes = 16
learning_rate = 0.01
epochs = 100

# Step 2: Feature Extraction
def extract_features(image_path):
    img = Image.open(image_path).resize(image_size)
    img_array = np.array(img.convert('L'))  # Convert to grayscale
    return img_array.flatten()

# Step 4: Linear Regression Model
def initialize_weights(input_size, output_size):
    w = np.random.randn(input_size, output_size)
    b = np.zeros((1, output_size))
    return w, b

def linear_regression(X, w, b):
    return np.dot(X, w) + b

def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# Load the training data (assuming you have lists of image file paths and labels)
train_image_paths = ["\\training Images"]  # List of file paths to training images
train_labels = ["\\TrainLabels.txt"]       # List of corresponding labels

# Load the testing data (assuming you have lists of image file paths and labels)
test_image_paths = ["\\testing Images"]   # List of file paths to testing images
test_labels = ["\\TestLabels.txt"]        # List of corresponding labels

# Step 5: Training
input_size = image_size[0] * image_size[1]
w, b = initialize_weights(input_size, num_classes)

for epoch in range(epochs):
    for i, image_path in enumerate(train_image_paths):
        # Load and preprocess the image
        features = extract_features(image_path)
        
        # Forward pass
        logits = linear_regression(features, w, b)
        
        # Convert logits to probabilities (e.g., using softmax)
        probabilities = ...  # Implement softmax function
        
        # Compute the loss
        loss = mean_squared_error(probabilities, train_labels[i])
        
        # Backpropagation and parameter update
        gradient = 2 * (probabilities - train_labels[i])
        w_gradient = np.outer(features, gradient)
        b_gradient = gradient
        w -= learning_rate * w_gradient
        b -= learning_rate * b_gradient

# Step 6: Testing
correct_predictions = 0
total_predictions = len(test_image_paths)

for i, image_path in enumerate(test_image_paths):
    # Load and preprocess the image
    features = extract_features(image_path)
    
    # Forward pass
    logits = linear_regression(features, w, b)
    
    # Convert logits to probabilities (e.g., using softmax)
    probabilities = ...  # Implement softmax function
    
    # Predicted class
    predicted_class = np.argmax(probabilities)
    
    # Compare with the true label
    if predicted_class == test_labels[i]:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions
print("Accuracy:", accuracy)