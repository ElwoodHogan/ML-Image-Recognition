import numpy as np
from PIL import Image
import os, shutil, sys, time

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

        #print("Done batching, Time: " + str(time.time() - start))

        # Forward pass.
        logits = np.dot(x_batch_flat, weights) + bias
        #print("Done Logits, Time: " + str(time.time() - start))

        exp_logits = np.exp(logits)

        #print("Done EXP Logits, Time: " + str(time.time() - start))

        predicted_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        #print("Done predicted probs, Time: " + str(time.time() - start))

        #print("Done Passing, Time: " + str(time.time() - start))

        # Compute the categorical cross-entropy loss.
        loss = -np.mean(np.log(predicted_probs[range(len(y_batch)), y_batch] + 1e-10))

        #print("Done computing loss, Time: " + str(time.time() - start))

        # Compute gradients.
        dscores = predicted_probs
        dscores[range(len(y_batch)), y_batch] -= 1
        dscores /= len(y_batch)

        #print("Done computing gradients, Time: " + str(time.time() - start))

        dw = np.dot(x_batch_flat.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)

        #print("Done computing sums, Time: " + str(time.time() - start))

        # Update weights and bias.
        weights -= learning_rate * dw
        bias = bias - (learning_rate * db)

        print(f'%done {((i+1)*batch_size)/total}')
        print("Time: " + str(time.time() - start))

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