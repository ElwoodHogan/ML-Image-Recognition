import os
import numpy as np
from sklearn import svm
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV

print("Loading training dataset...")
subset = load_dataset("aharley/rvl_cdip", split="train[0:120000]", cache_dir='F:/huggingface_rvlcdip')
print("Training dataset loaded.")

flattened_image_array = []
target = []

for picture in subset:
    img = picture['image'].resize((120,120))
    img_array = np.asarray(img, dtype=np.float32)
    img_array /= 255.0
    flattened_image_array.append(img_array.flatten())
    target.append(picture['label'])

print(len(flattened_image_array))

# param_grid = {'C': [0.1,1, 10, 100],
#               'gamma': [1,0.1,0.01,0.001],
#               'kernel': ['rbf', 'poly', 'sigmoid']}
# grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)
# grid.fit(flattened_image_array ,target)
# print(grid.best_estimator_)

classifier = svm.SVC(kernel='rbf',C=1,gamma=0.001)

print('Training classifier.')
classifier.fit(flattened_image_array, target)

print('Loading testing/validation dataset...')
test_dataset = load_dataset("aharley/rvl_cdip", split="test[0:20000]", cache_dir='F:/huggingface_rvlcdip')
testing = test_dataset.select([i for i in range(len(test_dataset)) if i != 33669])
print('Testing dataset loaded.')

flattened_testing_images = []
testing_labels = []

for picture in testing:
    testing_img = picture['image'].resize((120,120))
    testing_img_array = np.asarray(testing_img, dtype=np.float32)
    testing_img_array /= 255.0
    flattened_testing_images.append(testing_img_array.flatten())
    testing_labels.append(picture['label'])

print('Running predictions using the classifier...')
predictions = classifier.predict(flattened_testing_images)

accuracy = accuracy_score(testing_labels, predictions) * 100
print('The accuracy of the model is: {:.2f}'.format(accuracy))
precision = precision_score(testing_labels, predictions, average='weighted') * 100
print('The precision of the model is: {:.2f}'.format(precision))
recall = recall_score(testing_labels, predictions, average='weighted') * 100
print('The recall of the model is: {:.2f}'.format(recall))

