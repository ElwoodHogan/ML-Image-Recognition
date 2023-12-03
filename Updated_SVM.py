import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import skimage
import PIL
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV

print("Loading dataset...")
dataset = load_dataset("aharley/rvl_cdip", split="train")
subset = load_dataset("aharley/rvl_cdip", split="train[0:100000]")
print(dataset[10])
print(len(subset))
print(subset[0])
flattened_image_array = []
target = []

for picture in subset:
    img = picture['image'].resize((180,180))
    img_array = np.asarray(img, dtype=np.float32)
    img_array /= 255.0
    flattened_image_array.append(img_array.flatten())
    target.append(picture['label'])

print(len(flattened_image_array))
#print(flattened_image_array)
#print(target)


# param_grid = {'C': [0.1,1, 10, 100],
#               'gamma': [1,0.1,0.01,0.001],
#               'kernel': ['rbf', 'poly', 'sigmoid']}
# grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)
# grid.fit(flattened_image_array ,target)
# print(grid.best_estimator_)
sgd_classifier = SGDClassifier(random_state=50, max_iter=1000)
classifier = svm.SVC(kernel='rbf',C=10,gamma=0.001)

sgd_classifier.fit(flattened_image_array, target)
#classifier.fit(flattened_image_array, target)

testing = load_dataset("aharley/rvl_cdip", split="test[0:8000]")
flattened_testing_images = []
testing_labels = []

for picture in testing:
    testing_img = picture['image'].resize((180,180))
    testing_img_array = np.asarray(testing_img, dtype=np.float32)
    testing_img_array /= 255.0
    flattened_testing_images.append(testing_img_array.flatten())
    testing_labels.append(picture['label'])

#predictions = classifier.predict(flattened_testing_images)
predictions = sgd_classifier.predict(flattened_testing_images)
print(predictions)
#print(testing_labels)
accuracy = accuracy_score(testing_labels, predictions)
print(accuracy)
precision = precision_score(testing_labels, predictions, average='weighted')
print(precision)
recall = recall_score(testing_labels, predictions, average='weighted')