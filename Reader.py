import numpy as np
from PIL import Image
import os, shutil, sys
#THIS FILE IS FOR READING THE CONTENTS OF THE IMAGES AND FORMATTING THEM INTO ACCESSIBLE DATA
#For example, turning all the images into 1000x1000 images, and saving them to a new folder

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
pathTotrainingFolder = os.path.join(script_directory, trainingFolder)
pathTotestingFolder = os.path.join(script_directory, testingFolder)
pathTovalidationFolder = os.path.join(script_directory, validationFolder)

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


ClearFolder(pathTotrainingFolder)
ClearFolder(pathTotestingFolder)
ClearFolder(pathTovalidationFolder)



def findImageInData(imageDirec):
    with open("Image Database\labels\\test.txt") as file:
        for line in file:
            if imageDirec in line:
                #print("test")
                #print(line.rstrip()[-1])
                #return "Test " + line.rstrip()[-1], 0
                return line.rstrip()[-1], 0
            
    with open("Image Database\labels\\train.txt") as file:
        for line in file:
            if imageDirec in line:
                #print("train")
                #print(line.rstrip())
                #return "train " + line.rstrip()[-1], 1
                return line.rstrip()[-1], 1
            
    with open("Image Database\labels\\val.txt") as file:
        for line in file:
            if imageDirec in line:
                #print("valyee")
                #print(line.rstrip())
                #return "val " + line.rstrip()[-1], 2
                return line.rstrip()[-1], 2

def searchFiles():

    ImageIndex = 0

    NewTestFileName = 'TestLabels.txt'
    NewTrainFileName = 'TrainLabels.txt'
    NewValFileName = 'ValLabels.txt'

    PathToNewTestFileName = os.path.join(script_directory, NewTestFileName)
    PathToNewTrainFileNamee = os.path.join(script_directory, NewTrainFileName)
    PathToNewValFileName = os.path.join(script_directory, NewValFileName)

    testData = open(PathToNewTestFileName, 'w')
    trainData = open(PathToNewTrainFileNamee, 'w')
    valData = open(PathToNewValFileName, 'w')

    for subdir, dirs, files in os.walk('Image Database\images'):
        for file in files:
            fileDir = os.path.join(subdir, file)
            image = Image.open(fileDir)
            new_image = image.resize((1000, 1000))
            ImageName = "NewImage" + str(ImageIndex) + ".tif"

            #this raw directory will be used to check the file for weather its in the training, testing, or validation set
            rawDirec = fileDir.replace('Image Database\images\\', '')
            rawDirec = rawDirec.replace('\\', '/')

            Newline, dataType = findImageInData(rawDirec)
            
            
            #print(rawDirec)
            

            
            #0 == testing, 1 == training, 2 == validation
            match dataType:
                case 0: 
                    new_image.save(os.path.join(pathTotestingFolder, ImageName))
                    #testData.write(str(ImageIndex) + " " + Newline + "\n")
                    testData.write(str(ImageIndex) + " " + Newline + "\n")
                case 1: 
                    new_image.save(os.path.join(pathTotrainingFolder, ImageName))
                    #trainData.write(str(ImageIndex) + " " + Newline + "\n")
                case 2: 
                    new_image.save(os.path.join(pathTovalidationFolder, ImageName))
                    #valData.write(str(ImageIndex) + " " + Newline + "\n")

            
            ImageIndex += 1
            #return
            if(ImageIndex > 300): return

searchFiles()