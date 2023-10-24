import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset
from PIL import Image

print("Loading dataset...")
dataset = load_dataset("aharley/rvl_cdip")

# Select the ENTIRE dataset for training
N_SAMPLES = len(dataset['train'])  # This should be 320,000
print(f"Selecting a subset of {N_SAMPLES} samples for training...")
subset_train = dataset['train'].select(range(N_SAMPLES))

IMG_WIDTH = 64
IMG_HEIGHT = 64

def preprocess_data(example):
    """Resize, convert images into flattened arrays, normalize, and extract labels."""
    # Resize the image
    img = example['image'].resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.asarray(img, dtype=np.float32).flatten()
    img /= 255.0  # Normalize to [0, 1]
    label = example['label']
    return img, label

print("Preprocessing training data...")
X_train, y_train = zip(*[preprocess_data(example) for example in subset_train])
val_range = len(dataset['validation']) # Select how much of the validation set you want to use
# For validation, you can either use a subset or the full set. 
subset_val = dataset['validation'].select(range(val_range))  # Adjust val_range for faster testing/tuning
print("Preprocessing validation data...")
X_val, y_val = zip(*[preprocess_data(example) for example in subset_val])

BATCH_SIZE = 10000  # You can adjust this based on memory specs
N_EPOCHS = 10

# Calculate the number of iterations needed
n_batches_per_epoch = len(X_train) // BATCH_SIZE
max_iterations = N_EPOCHS * n_batches_per_epoch

# Create and train MLP classifier
l_rate = 0.00025
print("Creating and training the MLP classifier...")
clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=max_iterations, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1, batch_size=BATCH_SIZE,
                    learning_rate_init=l_rate, learning_rate='adaptive')

clf.fit(X_train, y_train)

# Validate the model
print("Validating the model...")
y_pred = clf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')

# Print out confusion matrix metrics
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
# Print out relevant MLP parameters from the run
print(f'Parameters: Image Size: {IMG_HEIGHT}px, Train Size: {N_SAMPLES}, Validation Size: {val_range}, Learning Rate: {l_rate}')