import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
import time

#VARIABLES
EPOCHS = 1
BATCH_SIZE = 5000
FILE_NAME = "dataset500.h5"

start_time = time.time()

# Set memory growth for GPU to minimize potential issues
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_data_from_h5(filename):
    with h5py.File(filename, 'r') as f:
        print(f"loading at {time.time() - start_time}")
        x_train = np.array(f['train_images'])
        print(f"Done loading x_train at {time.time() - start_time}")
        y_train = np.array(f['train_labels'])
        print(f"Done loading y_train at {time.time() - start_time}")
        x_test = np.array(f['test_images'])
        print(f"Done loading x_test at {time.time() - start_time}")
        y_test = np.array(f['test_labels'])
        print(f"Done loading y_test at {time.time() - start_time}")
        
    return x_train, y_train, x_test, y_test

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        batch_x = self.x[idxs]
        batch_y = self.y[idxs]

        batch_x = batch_x.astype('float32') / 255.0
        batch_x = batch_x.reshape(batch_x.shape[0], -1)

        return batch_x, batch_y

def train_logistic_regression_gpu(train_gen, validation_data, epochs=EPOCHS):
    n_samples, height, width = train_gen.x.shape
    n_features = height * width


    n_classes = len(np.unique(train_gen.y))
    
    # Define the model
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_gen, epochs=epochs, verbose=1, validation_data=validation_data)

filename = FILE_NAME

x_train, y_train, x_test, y_test = load_data_from_h5(filename)

# Preprocessing is done within the generator. So, we only need to create instances of it.
batch_size = BATCH_SIZE
train_gen = DataGenerator(x_train, y_train, batch_size)
val_gen = DataGenerator(x_test, y_test, batch_size)

train_logistic_regression_gpu(train_gen, val_gen)
