import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
import PIL as PIL
import random
import pickle


def safe_preprocess_data(example, img_width, img_height, random_rotate):
    try:
        return preprocess_data(example, img_width, img_height, random_rotate)
    except PIL.UnidentifiedImageError:
        print(f"Cannot identify image file for example: {example}")
        return None, None

"""Resize, convert images into flattened arrays, normalize, and extract labels."""
def preprocess_data(example, img_width, img_height, random_rotate):
    try:
        img = example['image'].resize((img_width, img_height))

        # Random rotation in 90-degree increments 
        if random_rotate:
            rotation_angle = random.choice([0, 90, 180, 270])
            img = img.rotate(rotation_angle)

        img = np.asarray(img, dtype=np.float32).flatten()
        img /= 255.0  # Normalize
        label = example['label']
        return img, label
    except PIL.UnidentifiedImageError:
        print(f"Cannot identify image file for example: {example}")
        return None, None

def preprocess_for_gridsearch(dataset, img_width, img_height, random_rotate):
    processed_data = [safe_preprocess_data(example, img_width, img_height, random_rotate) for example in dataset]
    return zip(*[data for data in processed_data if data[0] is not None])

def batch_generator(dataset, img_width, img_height, batch_size=10000, skip_index=None):
    for i in range(0, len(dataset), batch_size):
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        if skip_index and skip_index in batch_indices:
            batch_indices = [idx for idx in batch_indices if idx != skip_index]
        batch_data = dataset.select(batch_indices)
        processed_batch = [safe_preprocess_data(example, img_width, img_height, random_rotate) for example in batch_data]
        X_batch, y_batch = zip(*[data for data in processed_batch if data[0] is not None])
        yield np.array(X_batch), np.array(y_batch)

def evaluate_gridsearch(clf, X_test, y_test, params, img_size, file_name):
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='weighted')
    test_recall = recall_score(y_test, y_pred, average='weighted')
    test_f1 = f1_score(y_test, y_pred, average='weighted')

    output_str = (
        f"Image Size: {img_size[0]}x{img_size[1]}\n"
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
        f"Test Precision: {test_precision * 100:.2f}%\n"
        f"Test Recall: {test_recall * 100:.2f}%\n"
        f"Test F1 Score: {test_f1 * 100:.2f}%\n"
        f"Model Parameters: {params}\n\n"
    )

    print(output_str)

    with open(file_name, 'a') as file:
        file.write(output_str)

if __name__ == '__main__':

    print("Loading dataset...")
    dataset = load_dataset("aharley/rvl_cdip")

    # Select the ENTIRE dataset for training
    N_SAMPLES = (len(dataset['train']) // 16) # Can adjust the int for varying train subset sizes. 1 = Entire subset

    # These numbers represent the resolution we will load the images at.
    # For resolutions >128x128 I recommend enabling the batch generator
    img_width= 32      # You can adjust this based on memory specs
    img_height = 32    # You can adjust this based on memory specs

    # These numbers are for the classifier parameters themselves
    BATCH_SIZE = 2500   # You can adjust this based on memory specs
    N_EPOCHS = 2       # Adjust based on how many epochs you would like to perform
    l_rate = 0.000005   # Adjust the model's initial learning rate

    SKIP_INDEX = 33669  # DO NOT TOUCH. This image in the test subset is corrupted. We must skip it or we run into issues

    # This is the amount of images from the dataset will be loaded into memory
    # Adjust based on memory specs
    DATA_SUBSET_BATCH_SIZE = 10000 
    enable_dataset_batching = False # Enable for processing data subsets in batches for higher resolutions

    # These variable are related to using Grid Search for our model
    enable_grid_search = False # Enable for grid search for simultaneous multi-parameter model training and evaluation
    N_SAMPLES_GRID_SEARCH = 20000 # Set this for the amount of images you want to load for Gridsearch
    # Adjust variables to explore multiple configurations in parallel
    parameter_space = {
        'hidden_layer_sizes': [(128, 64), (256, 128)],
        'learning_rate_init': [0.0001, 0.00001],
        'batch_size': [5000, 10000],
        'max_iter': [25, 50]
    }
    # For grid search you must set this additional parameter. A list size 1 tuple is valid
    image_sizes = [(32, 32)]

    random_rotate = True # Enable for randomly rotating images in 90° increments (0, 90, 180, 270)

    if enable_dataset_batching and enable_grid_search:
        print('Error: Can not enable batching and grid search together. Exiting.')
        exit()

    # Save model to a file
    model_filename = 'MLP_folder/experiment_results/mlp_classifier_model.pkl'

    test_dataset = load_dataset("aharley/rvl_cdip", split="test")
    subset_test = test_dataset.select([i for i in range(len(test_dataset)) if i != 33669])

    if enable_grid_search:

        subset_train = dataset['train'].select(range(N_SAMPLES_GRID_SEARCH))
        val_range = len(dataset['validation']) // 4
        subset_val = dataset['validation'].select(range(val_range))
        test_range = (len(subset_test) // 4)
        print("Preprocessing Test Data...")
        processed_test_data = [safe_preprocess_data(example, img_width, img_height, random_rotate) for example in subset_test]
        log_file_name = 'MLP_folder/experiment_results/MLP_grid_search_performance_log.txt'
        model_performance = []

        for img_size in image_sizes:
            print('Preprocessing training, validation, and testing datasets...')
            img_width, img_height = img_size
            X_train, y_train = zip(*[preprocess_data(example, img_width, img_height, random_rotate) for example in subset_train])
            X_val, y_val = zip(*[preprocess_data(example, img_width, img_height, random_rotate) for example in subset_val])
            X_test, y_test = zip(*[data for data in processed_test_data if data[0] is not None])

            clf = MLPClassifier(random_state=1)
            grid = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3, scoring='f1_weighted', verbose=10)
            grid.fit(X_train, y_train)

            # Evaluate and log each model in the grid search
            print(f'Evaluating model with {img_size}px sized images...')
            for params in grid.cv_results_['params']:
                clf.set_params(**params)
                clf.fit(X_train, y_train)
                evaluate_gridsearch(clf, X_test, y_test, params, img_size, log_file_name)

                # Store the model and its performance for later comparison
                y_pred = clf.predict(X_test)
                test_f1 = f1_score(y_test, y_pred, average='weighted')
                model_performance.append((test_f1, clf, params, img_size))

        # Sort models based on F1 score and save the top 3 models
        top_models = sorted(model_performance, key=lambda x: x[0], reverse=True)[:3]
        for i, (score, model, params, img_size) in enumerate(top_models):
            with open(f'MLP_folder/experiment_results/top_model_{i+1}.pkl', 'wb') as file:
                pickle.dump(model, file)
        print("Grid search concluded. Find results in output .txt files.")
        exit()

    print(f"Selecting a subset of {N_SAMPLES} samples for training...")
    subset_train = dataset['train'].select(range(N_SAMPLES))

    if not enable_dataset_batching:
        print("Preprocessing Training Data...")
        X_train, y_train = zip(*[preprocess_data(example, img_width, img_height, random_rotate) for example in subset_train])

    # Validation data set initialzation
    val_range = (len(dataset['validation']) // 4)# Select how much of the validation set you want to use
    # For validation, you can either use a subset or the full set. 
    subset_val = dataset['validation'].select(range(val_range))  # Adjust val_range for faster testing/tuning
    print("Preprocessing Validation Data...")
    X_val, y_val = zip(*[preprocess_data(example, img_width, img_height, random_rotate) for example in subset_val])

    # Test set initialization
    test_range = (len(subset_test) // 4)
    print("Preprocessing Test Data...")
    processed_test_data = [safe_preprocess_data(example, img_width, img_height, random_rotate) for example in subset_test]
    X_test, y_test = zip(*[data for data in processed_test_data if data[0] is not None])


    # Calculate the number of iterations needed to run model for specified epochs
    n_batches_per_epoch = len(subset_train) // BATCH_SIZE
    max_iterations = N_EPOCHS * n_batches_per_epoch

    # Create and train MLP classifier
    print("Creating and training the MLP classifier...")
    clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=max_iterations, alpha=1e-4,
                        solver='adam', verbose=10, random_state=1, batch_size=BATCH_SIZE,
                        learning_rate_init=l_rate, learning_rate='adaptive', warm_start=False, n_iter_no_change=50, early_stopping=False)

    if enable_dataset_batching:
        print('Dataset batching has been enabled...')
        print(f'Processing training data in batches of {DATA_SUBSET_BATCH_SIZE} images...')
        for epoch in range(N_EPOCHS):
            print(f"Epoch {epoch+1}/{N_EPOCHS}")
            for X_batch, y_batch in batch_generator(dataset['train'], img_width, img_height, DATA_SUBSET_BATCH_SIZE):
                clf.partial_fit(X_batch, y_batch, classes=np.unique(y_batch))
    else:
        clf.fit(X_train, y_train)
    # Validate the model
    print("Validating the model...")
    y_pred = clf.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_pred)
    val_precision = precision_score(y_val, y_pred, average='weighted')
    val_recall = recall_score(y_val, y_pred, average='weighted')
    val_f1 = f1_score(y_val, y_pred, average='weighted')

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
    output_str += (f'Parameters:\n\tImage Size: {img_height}x{img_width}px\n\tRandom Orientation Rotation: {random_rotate}\n\tNumber of Epochs: {N_EPOCHS}' 
                + f'\n\tTrain Subset Size: {N_SAMPLES}\n\tValidation Subset Size: {val_range}\n\tTest Subset Size: {test_range}\n')
    output_str += f'MLP Classifier Parameters:\n{params_str}\n'

    print(output_str)

    with open(model_filename, 'wb') as file:
        pickle.dump(clf, file)
    print(f"Model saved to {model_filename}")


