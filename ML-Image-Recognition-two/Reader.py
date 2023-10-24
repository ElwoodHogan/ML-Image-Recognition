import numpy as np
from PIL import Image
import os, shutil, sys
import time
#THIS FILE IS FOR READING THE CONTENTS OF THE IMAGES AND FORMATTING THEM INTO ACCESSIBLE DATA
#For example, turning all the images into 1000x1000 images, and saving them to a new folder

#How many images do you want to process?  Any number above 400,000 will be all the images
ImageLimit = 20000

#if you want to start from scratch, setting this to true will reset the progress file
resetProgress = True

#these are only applied if reseting
NewTestFileName = 'TestLabels.txt'
NewTrainFileName = 'TrainLabels.txt'
NewValFileName = 'ValLabels.txt'

#how often do you want to be shown percentage updates?  smaller == more updates
percentThreshholdStep = .03

start = time.time()

#deletes the contents of the folder
def ClearFolder(pathToFolder):
    folder = pathToFolder
    for filename in os.listdir(folder): 
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

trainingFolder = "training Images"
testingFolder = "testing Images"
validationFolder = "validation Images"

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

PathToNewTestFileName = os.path.join(script_directory, NewTestFileName)
PathToNewTrainFileNamee = os.path.join(script_directory, NewTrainFileName)
PathToNewValFileName = os.path.join(script_directory, NewValFileName)


pathTotrainingFolder = os.path.join(script_directory, trainingFolder)
pathTotestingFolder = os.path.join(script_directory, testingFolder)
pathTovalidationFolder = os.path.join(script_directory, validationFolder)

#Creates the folder if theyre not already present
try:
    os.mkdir(pathTotrainingFolder)
except Exception as e:
    True #if the folder is already created, do nothing

try:
    os.mkdir(pathTotestingFolder)
except Exception as e:
    True #if the folder is already created, do nothing

try:
    os.mkdir(pathTovalidationFolder)
except Exception as e:
    True #if the folder is already created, do nothing

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

def searchFiles():
    
    progressFile = open("progress.txt", "r+")
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
    
    for subdir, dirs, files in os.walk('Image Database\images'):
        for file in files:
            if ImageIndex < progress: 
                ImageIndex+=1
                continue

            fileDir = os.path.join(subdir, file)

            image = Image.open(fileDir)
            new_image = image.resize((1000, 1000))
            ImageName = "NewImage" + str(ImageIndex) + ".tif"

            #this raw directory will be used to check the file for weather its in the training, testing, or validation set
            rawDirec = fileDir.replace('Image Database\images\\', '')
            rawDirec = rawDirec.replace('\\', '/')

            Newline, dataType = findImageInData(rawDirec)
            
            #appends to the label files
            #0 == testing, 1 == training, 2 == validation
            match dataType:
                case 0: 
                    new_image.save(os.path.join(pathTotestingFolder, ImageName))
                    testData.write(str(Newline) + "\n")
                case 1: 
                    new_image.save(os.path.join(pathTotrainingFolder, ImageName))
                    trainData.write(str(Newline) + "\n")
                case 2: 
                    new_image.save(os.path.join(pathTovalidationFolder, ImageName))
                    valData.write(str(Newline) + "\n")

            
            ImageIndex += 1

            #printing percent done
            percentDone = round((ImageIndex-progress)/(ImageLimit), 2)
            if(percentDone > percentThreshhold): 
                toString = ('%.1f' % (percentDone*100)).replace('.0', '')
                print(toString + "% Percent done")
                percentThreshhold+=percentThreshholdStep
            #return
            if(ImageIndex >= ImageLimit + progress): 
                progressFile = open("progress.txt", "w")
                progressFile.write(str(ImageLimit + progress))
                progressFile.close()
                return

def ClearProgress():

    #deletes the contents of the folders
    ClearFolder(pathTotrainingFolder)
    ClearFolder(pathTotestingFolder)
    ClearFolder(pathTovalidationFolder)

    #deletes the contents of the labels
    testData = open(PathToNewTestFileName, 'w')
    trainData = open(PathToNewTrainFileNamee, 'w')
    valData = open(PathToNewValFileName, 'w')

    #saves 0 to the progress file
    progressFile = open("progress.txt", "w")
    progressFile.write("0")
    progressFile.close()

if resetProgress:
    ClearProgress()

searchFiles()

end = time.time()
print(end - start)