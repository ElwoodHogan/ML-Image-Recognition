import h5py
import os
import cv2
import numpy as np
from multiprocessing import Pool, current_process, cpu_count
import time
from multiprocessing import Manager

BASE_DIR = "Image Database"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
TARGET_WIDTH = 1000  # or any custom width you desire
TARGET_HEIGHT = 1000  # or any custom height you desire

start_time = time.time()

TEST_LIMIT = 400000
NEW_DATASET_NAME = "dataset1000.h5"
RESIZE_INTERPOLATION = cv2.INTER_LANCZOS4
BATCH_SIZE = 5000

def read_label_file(filename):
    index = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        paths, labels = [], []
        for line in lines:
            path, label = line.strip().split()
            paths.append(os.path.join(IMAGE_DIR, path))
            labels.append(int(label))
            index +=1
            if(index == TEST_LIMIT): break
        return paths, labels

def parse_file_paths(lines):
    paths = []
    labels = []
    for line in lines:
        path, label = line.strip().split(' ')
        paths.append(path)
        labels.append(int(label))
    return paths, labels

def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation = RESIZE_INTERPOLATION)
    return img

def worker(task, progress_counter, progress_lock):
    img_path, label_path, idx, total = task
    img = process_image(img_path)

    with progress_lock:
        progress_counter.value += 1
        progress = progress_counter.value
    if progress % BATCH_SIZE == 0:  # Adjust this number as required
        estimated_time_left = (total - progress) * (time.time() - start_time) / progress
        print(f"Processed {progress}/{total}. Estimated time left: {estimated_time_left:.2f} seconds")

    return img, label_path

def worker_wrapper(args):
    return worker(args[0], args[1], args[2])

def chunk_data(data, batch_size):
    """Split data into chunks of batch size."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def main():
    train_paths, train_labels = read_label_file(os.path.join(LABELS_DIR, 'train.txt'))
    test_paths, test_labels = read_label_file(os.path.join(LABELS_DIR, 'test.txt'))

    total_images = len(train_paths) + len(test_paths)
    print(total_images)

    tasks = []
    for idx, path in enumerate(train_paths):
        tasks.append((path, 'train.txt', idx, total_images))
    for idx, path in enumerate(test_paths):
        tasks.append((path, 'test.txt', idx + len(train_paths), total_images))

    with h5py.File(NEW_DATASET_NAME, 'w') as hf:
        train_images_dset = hf.create_dataset("train_images", (len(train_paths), TARGET_HEIGHT, TARGET_WIDTH), dtype='uint8')
        train_labels_dset = hf.create_dataset("train_labels", (len(train_labels),), dtype='int32')
        test_images_dset = hf.create_dataset("test_images", (len(test_paths), TARGET_HEIGHT, TARGET_WIDTH), dtype='uint8')
        test_labels_dset = hf.create_dataset("test_labels", (len(test_labels),), dtype='int32')

        with Manager() as manager:
            progress_counter = manager.Value('i', 0)
            progress_lock = manager.Lock()

            for batch in chunk_data(tasks, BATCH_SIZE):
                tasks_with_counter = [(task, progress_counter, progress_lock) for task in batch]

                with Pool(cpu_count()) as pool:
                    trainCount = 0
                    testCount = 0
                    for idx, (img, label_path) in enumerate(pool.imap_unordered(worker_wrapper, tasks_with_counter)):
                        if "train" in label_path:
                            train_images_dset[trainCount] = img
                            train_labels_dset[trainCount] = train_labels[trainCount]
                            trainCount+=1
                        elif "test" in label_path:
                            test_images_dset[testCount] = img
                            test_labels_dset[testCount] = test_labels[testCount]
                            testCount+=1

if __name__ == "__main__":
    main()
