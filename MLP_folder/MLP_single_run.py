import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
import PIL as PIL
import random
import joblib

random_rotate = False

def safe_preprocess_data(example):
    try:
        return preprocess_data(example)
    except PIL.UnidentifiedImageError:
        print(f"Cannot identify image file for example: {example}")
        return None, None

def preprocess_data(example):
    """Resize, convert images into flattened arrays, normalize, and extract labels."""
    try:
        img = example['image'].resize((IMG_WIDTH, IMG_HEIGHT))

        # UNCOMMENT FOR: Random rotation in 90-degree increments 
        # rotation_angle = random.choice([0, 90, 180, 270])
        # img = img.rotate(rotation_angle)
        # random_rotate = True

        img = np.asarray(img, dtype=np.float32).flatten()
        img /= 255.0  # Normalize
        label = example['label']
        return img, label
    except PIL.UnidentifiedImageError:
        print(f"Cannot identify image file for example: {example}")
        return None, None

def batch_generator(dataset, batch_size=10000, skip_index=None):
    for i in range(0, len(dataset), batch_size):
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        if skip_index and skip_index in batch_indices:
            batch_indices = [idx for idx in batch_indices if idx != skip_index]
        batch_data = dataset.select(batch_indices)
        processed_batch = [safe_preprocess_data(example) for example in batch_data]
        X_batch, y_batch = zip(*[data for data in processed_batch if data[0] is not None])
        yield np.array(X_batch), np.array(y_batch)

if __name__ == '__main__':
    print("Loading dataset...")
    dataset = load_dataset("aharley/rvl_cdip")
    test_dataset = load_dataset("aharley/rvl_cdip", split="test")
    subset_test = test_dataset.select([i for i in range(len(test_dataset)) if i != 33669])

    # Select the ENTIRE dataset for training
    N_SAMPLES = (len(dataset['train']) // 16)
    print(f"Selecting a subset of {N_SAMPLES} samples for training...")
    subset_train = dataset['train'].select(range(N_SAMPLES))

    IMG_WIDTH = 32 # You can adjust this based on memory specs
    IMG_HEIGHT = 32    # You can adjust this based on memory specs
    BATCH_SIZE = 2500   # You can adjust this based on memory specs
    N_EPOCHS = 10
    SKIP_INDEX = 33669  # This image in the test subset is corrupted. We must skip it or we run into issues

    print("Preprocessing Training Data...")
    X_train, y_train = zip(*[preprocess_data(example) for example in subset_train])

    # Validation data set initialzation
    val_range = (len(dataset['validation']))# Select how much of the validation set you want to use
    # For validation, you can either use a subset or the full set. 
    subset_val = dataset['validation'].select(range(val_range))  # Adjust val_range for faster testing/tuning
    print("Preprocessing Validation Data...")
    X_val, y_val = zip(*[preprocess_data(example) for example in subset_val])

    # Test set initialization
    test_range = len(subset_test)
    #subset_test = dataset['test'].select(range(test_range))
    print("Preprocessing Test Data...")
    processed_test_data = [safe_preprocess_data(example) for example in subset_test]
    X_test, y_test = zip(*[data for data in processed_test_data if data[0] is not None])


    # Calculate the number of iterations needed to run model for specified epochs
    n_batches_per_epoch = len(X_train) // BATCH_SIZE
    max_iterations = N_EPOCHS * n_batches_per_epoch

    # Create and train MLP classifier
    l_rate = 0.000005
    print("Creating and training the MLP classifier...")
    clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=max_iterations, alpha=1e-4,
                        solver='adam', verbose=10, random_state=1, batch_size=BATCH_SIZE,
                        learning_rate_init=l_rate, learning_rate='adaptive', warm_start=False, n_iter_no_change=50, early_stopping=False)

    clf.fit(X_train, y_train)

    # Validate the model
    print("Validating the model...")
    y_pred = clf.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_pred)
    val_precision = precision_score(y_val, y_pred, average='weighted')
    val_recall = recall_score(y_val, y_pred, average='weighted')
    val_f1 = f1_score(y_val, y_pred, average='weighted')

    # Save model to a file
    model_filename = 'MLP_folder/experiment_results/mlp_single_run_model.joblib'
    joblib.dump(clf, model_filename)
    print(f"Model saved to {model_filename}")

    params = clf.get_params()

    # Create a string to display these parameters
    params_str = "\n".join([f"\t{key}: {value}" for key, value in params.items()])

    # Create a string to hold the output
    output_str = ""
    output_str += f"Validation Accuracy: {val_accuracy * 100:.2f}%\n"
    output_str += f"Validation Precision: {val_precision * 100:.2f}%\n"
    output_str += f"Validation Recall: {val_recall * 100:.2f}%\n"
    output_str += f"Validation F1 Score: {val_f1 * 100:.2f}%\n"

    # Test the model
    y_pred = clf.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='weighted')
    test_recall = recall_score(y_test, y_pred, average='weighted')
    test_f1 = f1_score(y_test, y_pred, average='weighted')

    # Append test results to the string
    output_str += f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
    output_str += f"Test Precision: {test_precision * 100:.2f}%\n"
    output_str += f"Test Recall: {test_recall * 100:.2f}%\n"
    output_str += f"Test F1 Score: {test_f1 * 100:.2f}%\n"
    output_str += (f'Parameters:\n\tImage Size: {IMG_HEIGHT}x{IMG_WIDTH}px,\n\tRandom Orientation Rotation: {random_rotate}\n\tNumber of Epochs: {N_EPOCHS},' 
                + f'\n\tTrain Subset Size: {N_SAMPLES},\n\tValidation Subset Size: {val_range}\n\tTest Subset Size: {test_range}\n')
    output_str += f'MLP Classifier Parameters:\n{params_str}\n'

    print(output_str)


