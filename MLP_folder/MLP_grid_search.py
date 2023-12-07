import numpy as np
import PIL as PIL
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset
import pickle

def safe_preprocess_data(example, img_width, img_height):
    try:
        return preprocess_data(example, img_width, img_height)
    except PIL.UnidentifiedImageError:
        print(f"Cannot identify image file for example: {example}")
        return None, None

def preprocess_data(example, img_width, img_height):
    try:
        img = example['image'].resize((img_width, img_height))
        img = np.asarray(img, dtype=np.float32).flatten()
        img /= 255.0
        label = example['label']
        return img, label
    except PIL.UnidentifiedImageError:
        print(f"Cannot identify image file for example: {example}")
        return None, None

def preprocess_dataset(dataset, img_size):
    processed_data = [safe_preprocess_data(example, *img_size) for example in dataset]
    return zip(*[data for data in processed_data if data[0] is not None])

def evaluate_model(clf, X_test, y_test, params, img_size, file_name):
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
    test_dataset = load_dataset("aharley/rvl_cdip", split="test")
    subset_test = test_dataset.select([i for i in range(len(test_dataset)) if i != 33669])

    parameter_space = {
        'hidden_layer_sizes': [(256, 128), (512, 256), (1024, 512)],
        'learning_rate_init': [0.01, 0.001, 0.0001],
        'batch_size': [5000, 10000, 20000],
        'max_iter': [250, 500, 1000]
    }

    image_sizes = [(64, 64), (128, 128), (256, 256)]

    # Select the first 20000 images of dataset for training
    N_SAMPLES = 20000
    subset_train = dataset['train'].select(range(N_SAMPLES))
    val_range = len(dataset['validation']) // 2
    subset_val = dataset['validation'].select(range(val_range))

    log_file_name = 'MLP_folder/experiment_results/MLP_grid_search_performance_log.txt'
    model_performance = []

    for img_size in image_sizes:
        print('Preprocessing training, validation, and testing datasets...')
        X_train, y_train = preprocess_dataset(subset_train, img_size)
        X_val, y_val = preprocess_dataset(subset_val, img_size)
        X_test, y_test = preprocess_dataset(subset_test, img_size)

        clf = MLPClassifier(random_state=1)
        grid = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3, scoring='f1_weighted', verbose=10)
        grid.fit(X_train, y_train)

        # Evaluate and log each model in the grid search
        print(f'Evaluating model with {img_size}px sized images...')
        for params in grid.cv_results_['params']:
            clf.set_params(**params)
            clf.fit(X_train, y_train)
            evaluate_model(clf, X_test, y_test, params, img_size, log_file_name)

            # Store the model and its performance for later comparison
            y_pred = clf.predict(X_test)
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            model_performance.append((test_f1, clf, params, img_size))

    # Sort models based on F1 score and save the top 3 models
    top_models = sorted(model_performance, key=lambda x: x[0], reverse=True)[:3]
    for i, (score, model, params, img_size) in enumerate(top_models):
        with open(f'MLP_folder/experiment_results/top_model_{i+1}.pkl', 'wb') as file:
            pickle.dump(model, file)



