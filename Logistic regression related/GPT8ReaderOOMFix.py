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
TARGET_WIDTH = 512
TARGET_HEIGHT = 512

start_time = time.time()

TEST_LIMIT = 400000
NEW_DATASET_NAME = "dataset500.h5"
RESIZE_INTERPOLATION = cv2.INTER_LANCZOS4

def read_label_file(filename):
    with open(filename, 'r') as f:
        for index, line in enumerate(f):
            if index == TEST_LIMIT:
                break
            path, label = line.strip().split()
            yield os.path.join(IMAGE_DIR, path), int(label)

def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=RESIZE_INTERPOLATION)
    return img

def worker(task, progress_counter, progress_lock):
    img_path, label, idx, total = task
    img = process_image(img_path)

    with progress_lock:
        progress_counter.value += 1
        progress = progress_counter.value
    if progress % 1000 == 0:
        estimated_time_left = (total - progress) * (time.time() - start_time) / progress
        print(f"Processed {progress}/{total}. Estimated time left: {estimated_time_left:.2f} seconds")

    return img, label

def worker_wrapper(args):
    return worker(args[0], args[1], args[2])

def main():
    train_data = list(read_label_file(os.path.join(LABELS_DIR, 'train.txt')))
    test_data = list(read_label_file(os.path.join(LABELS_DIR, 'test.txt')))

    total_images = len(train_data) + len(test_data)
    print(total_images)

    tasks = [(path, label, idx, total_images) for idx, (path, label) in enumerate(train_data + test_data)]

    with h5py.File(NEW_DATASET_NAME, 'w') as hf:
        train_images_dset = hf.create_dataset("train_images", (len(train_data), TARGET_HEIGHT, TARGET_WIDTH), dtype='uint8')
        train_labels_dset = hf.create_dataset("train_labels", (len(train_data),), dtype='int32')
        test_images_dset = hf.create_dataset("test_images", (len(test_data), TARGET_HEIGHT, TARGET_WIDTH), dtype='uint8')
        test_labels_dset = hf.create_dataset("test_labels", (len(test_data),), dtype='int32')

        with Manager() as manager:
            progress_counter = manager.Value('i', 0)
            progress_lock = manager.Lock()
            tasks_with_counter = [(task, progress_counter, progress_lock) for task in tasks]

            # Use half the cores to reduce memory consumption
            with Pool(cpu_count() // 2) as pool:
                trainCount, testCount = 0, 0
                for idx, (img, label) in enumerate(pool.imap_unordered(worker_wrapper, tasks_with_counter)):
                    if idx < len(train_data):
                        train_images_dset[trainCount] = img
                        train_labels_dset[trainCount] = label
                        trainCount += 1
                    else:
                        test_images_dset[testCount] = img
                        test_labels_dset[testCount] = label
                        testCount += 1

if __name__ == "__main__":
    main()
