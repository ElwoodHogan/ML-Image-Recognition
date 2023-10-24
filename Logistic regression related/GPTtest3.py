import numpy as np
from PIL import Image
import os, shutil, sys, time
import tensorflow as tf

# Generate some random data for demonstration purposes.
# In practice, you should load your dataset.
num_classes = 16

start = time.time()

Xtrain = []
Ytrain = np.loadtxt("TrainLabels.txt", dtype=int)

Xtest = []
Ytest = np.loadtxt("TestLabels.txt", dtype=int)

def CreateTrainingArrays():
    for subdir, dirs, files in os.walk('training Images'):
        for file in files:
            fileDir = os.path.join(subdir, file)
            img = Image.open(fileDir)
            img_as_np = np.asarray(img)
            nx, ny = img_as_np.shape
            flattentedIMG = img_as_np.reshape((nx*ny))
            Xtrain.append(flattentedIMG)

def CreateTestingArrays():
    for subdir, dirs, files in os.walk('testing Images'):
        for file in files:
            fileDir = os.path.join(subdir, file)
            img = Image.open(fileDir)
            img_as_np = np.asarray(img)
            nx, ny = img_as_np.shape
            flattentedIMG = img_as_np.reshape((nx*ny))
            Xtest.append(flattentedIMG)

print("Creating initial arrays, Time: " + str(time.time() - start))
CreateTrainingArrays()
CreateTestingArrays()

Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)

Xtrain = Xtrain.astype(np.float32)
Xtest = Xtest.astype(np.float32)
print("Done creating initial arrays, Time: " + str(time.time() - start))

total = len(Xtrain)
print(total)
# Initialize weights and bias.
input_dim = len(Xtrain[1])
output_dim = num_classes
learning_rate = 0.01
epochs = 1
batch_size = 8000

weights = np.zeros((input_dim, output_dim))
bias = np.zeros(output_dim)

print("Training, Time: " + str(time.time() - start))

# Training loop using gradient descent with mini-batches.
for epoch in range(epochs):
    for i in range(0, len(Xtrain), batch_size):
        x_batch_flat = Xtrain[i:i+batch_size]
        y_batch = Ytrain[i:i+batch_size]

        with tf.GradientTape() as tape:
            logits = tf.matmul(x_batch_flat, weights) + bias
            predicted_probs = tf.nn.softmax(logits)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_batch, predicted_probs))

        grads = tape.gradient(loss, [weights, bias])
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        optimizer.apply_gradients(zip(grads, [weights, bias]))

    # Print loss for monitoring.
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss}')
        print("Time: " + str(time.time() - start))

# Evaluation on the test set.
logits_test = np.dot(Xtest, weights) + bias
exp_logits_test = np.exp(logits_test)
predicted_probs_test = exp_logits_test / np.sum(exp_logits_test, axis=1, keepdims=True)
predicted_labels_test = np.argmax(predicted_probs_test, axis=1)
accuracy = np.mean(predicted_labels_test == Ytest)
print(f'Test accuracy: {accuracy}')
print("Time: " + str(time.time() - start))