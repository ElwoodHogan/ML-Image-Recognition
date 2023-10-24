import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
import time
import gc
from keras.models import load_model

# VARIABLES
EPOCHS = 5
BATCH_SIZE = 5000
FILE_NAME = "dataset500.h5"
CHUNK_SIZE = 20000
MODEL_NAME = 'LogisticsModel.h5'

#if true, will load the model and test it rather than create a new one
TEST_MODEL = True

#if the program fails at a specific chunk due to memory allocation error, it will print which chunk that is
#you can run the program from that chunk by putting the number here
GOTO_CHUNK = 9

# Initialize timer
start_time = time.time()



# Check for GPU (this will print "0" if no compatible GPU is found)
print(f"{len(tf.config.list_physical_devices('GPU'))} GPUs found")

def load_data_in_chunks_from_h5(filename, chunk_size, data_type='train'):
    with h5py.File(filename, 'r') as f:
        total_size = f[f'{data_type}_images'].shape[0]
        chunks = int(np.ceil(total_size / chunk_size))
        
        for i in range(chunks):
            if TEST_MODEL == False:
                if GOTO_CHUNK != -1:
                    if i < GOTO_CHUNK:
                        continue
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_size)
            print(f"Loading data from index {start_idx} to {end_idx}")
            
            #NORMALIZE BY DIVIDING BY 255!!!  NOT DOING THIS SCREWS YOUR RUNTIME!
            x_chunk = np.array(f[f'{data_type}_images'][start_idx:end_idx]).astype('float16') / 255.0  # float16
            y_chunk = np.array(f[f'{data_type}_labels'][start_idx:end_idx])
            yield x_chunk, y_chunk

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.x[idxs], self.y[idxs]

def train_logistic_regression_gpu(filename, epochs=EPOCHS):
    n_classes = 0
    model = None  # Initialize model to None
    chunk_index = 0
    try:
        for x_chunk, y_chunk in load_data_in_chunks_from_h5(filename, CHUNK_SIZE, 'train'):
            chunk_index+=1

            if chunk_index == 1:  # Create the model only once
                if GOTO_CHUNK != -1:
                    print("loading model...")
                    model = load_model(MODEL_NAME)
                else:
                    print("creating model...")
                    n_classes = len(np.unique(y_chunk))
                    n_samples, height, width = x_chunk.shape
                    # Define the model
                    model = tf.keras.Sequential([
                        tf.keras.layers.Flatten(input_shape=(height, width)),
                        tf.keras.layers.Dense(n_classes, activation='softmax')
                    ])

                    model.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])

            train_gen_chunk = DataGenerator(x_chunk, y_chunk, BATCH_SIZE)  # Reduced batch size
            
            # Fit the model on the current chunk of data
            model.fit(train_gen_chunk, epochs=epochs, verbose=1)
            
            # Clear memory
            del x_chunk, y_chunk, train_gen_chunk
            gc.collect()  # Explicitly free memory

            print(f"saving model on chunk #{chunk_index}...")
            model.save(MODEL_NAME)
            print("Done.")
    except:
        gc.collect()  # Explicitly free memory
        print(f"Failed at chunk {chunk_index}")
    
    print("Saving model...")
    model.save(MODEL_NAME)
    print("Done.")
    return model


# Main function
if __name__ == '__main__':
    

    if TEST_MODEL:
        print("testing model...")
        model = load_model(MODEL_NAME)
        test_chunks = load_data_in_chunks_from_h5(FILE_NAME, 40000, 'test')
        test_x, test_y = next(test_chunks) 
        test_gen = DataGenerator(test_x, test_y, 40000)
        loss, accuracy = model.evaluate(test_gen, verbose = 1)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    else:
        train_logistic_regression_gpu(FILE_NAME)

