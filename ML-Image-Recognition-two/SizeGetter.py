import numpy as np
from PIL import Image
import os, shutil, sys
import time


def searchFiles():
    
    smallestX = [2000,2000]
    smallestY = [2000,2000]
    biggestX= [0,0]
    biggestY= [0,0]
    biggerThan1000 = 0

    index = 0
    for subdir, dirs, files in os.walk('Image Database\images'):
        for file in files:
            #print("yo")

            index+=1
            
            fileDir = os.path.join(subdir, file)

            

            image = Image.open(fileDir)
            #print(image.size)

            if(image.size[0] > 1000 or image.size[1] > 1000):
                biggerThan1000 +=1 
                print(str(biggerThan1000) + " " + str(index))

            if(image.size[0] < smallestX[0]):
                smallestX = image.size
                print("smallestX: " + str(smallestX) + fileDir)
            
            if(image.size[1] < smallestY[1]):
                smallestY = image.size
                print("smallestY: " + str(smallestY) + fileDir)
            
            if(image.size[0] > biggestX[0]):
                biggestX = image.size
                print("biggestX: " + str(biggestX) + fileDir)
            
            if(image.size[1] > biggestY[1]):
                biggestY = image.size
                print("biggestY: " + str(biggestY) + fileDir)
            

searchFiles()      