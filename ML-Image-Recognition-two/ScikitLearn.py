from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from PIL import Image
import os, shutil, sys, time
import numpy as np
from sklearn import svm
import time

start = time.time()

Xtrain = [] 
Ytrain = np.loadtxt("TrainLabels.txt", dtype=int)

Xtest = []
Ytest = np.loadtxt("TestLabels.txt", dtype=int)
#print(File_data)
#print(File_data[4])

def CreateTrainingArrays():
    for subdir, dirs, files in os.walk('training Images'):
        for file in files:
            fileDir = os.path.join(subdir, file)
            img = Image.open(fileDir)
            img_as_np = np.asarray(img)
            Xtrain.append(img_as_np)

def CreateTestingArrays():
    for subdir, dirs, files in os.walk('testing Images'):
        for file in files:
            fileDir = os.path.join(subdir, file)
            img = Image.open(fileDir)
            img_as_np = np.asarray(img)
            Xtest.append(img_as_np)

print("Creating initial arrays")
CreateTrainingArrays()
CreateTestingArrays()


#scikit only accepts 2d arrays, so we must first converty the 1000,1000 arrays into a 1000000 array
print("Formatting Training")
formattedX = []
for x in range(len(Xtrain)):
    nx, ny = Xtrain[x].shape
    formattedX.append(Xtrain[x].reshape((nx*ny)))

print("Formatting Testing")
formattedTrainX = []
for x in range(len(Xtest)):
    nx, ny = Xtest[x].shape
    formattedTrainX.append(Xtest[x].reshape((nx*ny)))

print("Pre-trainTime: " + str(time.time() - start))


startSVM = time.time()


print("Training SVM")
clf = svm.SVC()
clf.fit(formattedX, Ytrain)

print("Testing SVM")
y_pred = clf.predict(formattedTrainX)
print('svm Percentage correct: ', 100*np.sum(y_pred == Ytest)/len(Ytest))

print("Training and testing SVM: " + str(time.time() - startSVM))
'''

'''
startSGD = time.time()

print("Training SGD")
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(formattedX, Ytrain)

print("Testing SGD")
y_pred = sgd_clf.predict(formattedTrainX)
print('SGD Percentage correct: ', 100*np.sum(y_pred == Ytest)/len(Ytest))

print("Training and testing SGD: " + str(time.time() - startSGD))
print("Total Time: " + str(time.time() - start))
