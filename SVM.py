import os
import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
import skimage

# Build path
current_directory = os.getcwd()
subdirectory = "training Images"
subdirectoryA = "testing Images"

training_images_path = os.path.join(current_directory, subdirectory)
testing_images_path = os.path.join(current_directory, subdirectoryA)

# Create target list
train_file = "TrainLabels.txt"
labels = []
with open(train_file, 'r') as file:
    labels = file.readlines()

labels = [label.strip() for label in labels]
print(labels)

# Create target list for testing
test_file = "TestLabels.txt"
labels_test = []
with open(test_file, 'r') as file_test:
    labels_test = file_test.readlines()

labels_test = [label.strip() for label in labels_test]
print(labels_test)

# Create image array and label array
flattened_image_array = []
for file in os.listdir(training_images_path):
    image = skimage.io.imread(os.path.join(training_images_path, file))
    resized_image = skimage.transform.resize(image, (200, 200))
    image_array = np.array(resized_image)
    flattened_image = image_array.flatten()
    flattened_image_array.append(flattened_image)

images_array = np.array(flattened_image_array)

labels_array = np.array(labels)

df = pd.DataFrame(images_array)
df['Target'] = labels_array
print(df.shape)

# input data
x = df.iloc[:, :-1]
# output data
y = df.iloc[:, -1]

# train model
svm = svm.SVC(kernel='rbf', gamma=1.0)
svm.fit(x, y)

# Create testing data
flattened_image_array_testing = []
for file in os.listdir(testing_images_path):
    image_test = skimage.io.imread(os.path.join(testing_images_path, file))
    resized_image_test = skimage.transform.resize(image_test, (200, 200))
    image_array_test = np.array(resized_image_test)
    flattened_image_test = image_array_test.flatten()
    flattened_image_array_testing.append(flattened_image_test)

images_array_test = np.array(flattened_image_array_testing)
labels_array_test = np.array(labels_test)
df_test = pd.DataFrame(images_array_test)
df_test['Target'] = labels_array_test
print(df_test.shape)

# input data
x_test = df_test.iloc[:, :-1]

# output data
y_test = df_test.iloc[:, -1]

predictions = svm.predict(x_test)
print(predictions)
print(y_test)
accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
print(accuracy)
print(sklearn.metrics.classification_report(y_test, predictions))
