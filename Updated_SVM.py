import os
import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
import skimage
from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset("aharley/rvl_cdip")
training_data = dataset['train'].select(range(100))

print(training_data[:-1])

