import numpy as np
from PIL import Image
import os, shutil, sys
import time
import cv2
from joblib import Parallel, delayed

#THIS FILE IS FOR READING THE CONTENTS OF THE IMAGES AND FORMATTING THEM INTO ACCESSIBLE DATA
#For example, turning all the images into 1000x1000 images, and saving them to a new File

#How many images do you want to process?  Any number above 400,000 will be all the images
ImageLimit = 20

#if you want to start from scratch, setting this to true will reset the progress file
resetProgress = True

#these are only applied if reseting
NewTestFileName = 'TestLabels250.txt'
NewTrainFileName = 'TrainLabels250.txt'
NewValFileName = 'ValLabels250.txt'

ProgressFileName = "progress250.txt"

#image text names
trainingImages = "training Images File.txt"
testingImages = "testing Images File.txt"
validationImages = "validation Images File.txt"

#how often do you want to be shown percentage updates?  smaller == more updates
percentThreshholdStep = .03

NewImageSize = 250

#this number should equal your CPUs physical core count
ParrallelJobs = 16
n_jobs = -1  # Use all available cores. Adjust if you want to limit the number of cores.

root_dir = 'Image Database\images'

start = time.time()


script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

PathToNewTestFileName = os.path.join(script_directory, NewTestFileName)
PathToNewTrainFileNamee = os.path.join(script_directory, NewTrainFileName)
PathToNewValFileName = os.path.join(script_directory, NewValFileName)


pathTotrainingFile = os.path.join(script_directory, trainingImages)
pathTotestingFile = os.path.join(script_directory, testingImages)
pathTovalidationFile = os.path.join(script_directory, validationImages)

def process_image(image_file):
    # Load the image using OpenCV
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)  # Remove cv2.IMREAD_GRAYSCALE for color images
    # Resize it
    img_resized = cv2.resize(img, (NewImageSize, NewImageSize))
    # Flatten and return
    return img_resized.flatten()

def findImageInData(imageDirec):
    with open("Image Database\labels\\test.txt") as file:
        for line in file:
            if imageDirec in line:
                return line.rstrip()[-1], 0
            
    with open("Image Database\labels\\train.txt") as file:
        for line in file:
            if imageDirec in line:
                return line.rstrip()[-1], 1
            
    with open("Image Database\labels\\val.txt") as file:
        for line in file:
            if imageDirec in line:
                return line.rstrip()[-1], 2
    
    print("Dat not found!")

def searchFiles():
    
    progressFile = open(ProgressFileName, "r+")
    progress = int(progressFile.readline())
    progressFile.close()
    print(progress)

    percentThreshhold = percentThreshholdStep
    percentDone = 0
    ImageIndex = 0

    #opens the labels in append mode
    testData = open(PathToNewTestFileName, 'a')
    trainData = open(PathToNewTrainFileNamee, 'a')
    valData = open(PathToNewValFileName, 'a')

    for subdir, _, files in os.walk(root_dir):
                # Full paths of images in the current directory
                image_files = [os.path.join(subdir, file) for file in files]

                # Parallel processing of images in the current directory
                n_jobs = -1  # Use all available cores. Adjust if you want to limit the number of cores.
                images_1d_list = Parallel(n_jobs=n_jobs)(delayed(process_image)(image_file) for image_file in image_files)
                for image_file, img_arr in zip(image_files, images_1d_list):
                    rawDirectory = image_file.replace('Image Database\images\\', '')
                    rawDirectory = rawDirectory.replace('\\', '/')
                    Newline, dataType = findImageInData(rawDirectory)

                    # Determine the file to append based on dataType
                    if dataType == 0:  # testing
                        save_path = pathTotrainingFile
                        with open(PathToNewTestFileName, 'a') as f:
                            f.write(str(Newline) + "\n")
                    elif dataType == 1:  # training
                        save_path = pathTotestingFile
                        with open(PathToNewTrainFileNamee, 'a') as f:
                            f.write(str(Newline) + "\n")
                    elif dataType == 2:  # validation
                        save_path = pathTovalidationFile
                        with open(PathToNewValFileName, 'a') as f:
                            f.write(str(Newline) + "\n")
                    else:
                        raise ValueError(f"Unexpected dataType value: {dataType}")

                    # Append the 1D array to the respective text file
                    with open(save_path, 'a') as f:
                        img_str = ' '.join(map(str, img_arr))
                        f.write(img_str + '\n# Array end\n')


def ClearProgress():
    #deletes the contents of the labels
    testData = open(PathToNewTestFileName, 'w')
    trainData = open(PathToNewTrainFileNamee, 'w')
    valData = open(PathToNewValFileName, 'w')

    testImages = open(pathTotrainingFile, 'w')
    trainImages = open(pathTotestingFile, 'w')
    valImages = open(pathTovalidationFile, 'w')

    #saves 0 to the progress file
    progressFile = open(ProgressFileName, "w")
    progressFile.write("0")
    progressFile.close()

if resetProgress:
    ClearProgress()

searchFiles()

end = time.time()
print(end - start)

