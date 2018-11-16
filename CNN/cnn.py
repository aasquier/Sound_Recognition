import sys
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
sys.path.insert(0, '../')
from extractFeatures import readDataFile


#Data files
AUDIO_DATA      = "../data/audioData.csv"
IMAGE_DATA      = "../data/imageData.csv"
AUDIO_DATA_NORM = "../data/audioDataNormalized.csv"
IMAGE_DATA_NORM = "../data/imageDataNormalized.csv"

#Hyper Parameters
BATCH_SIZE = 128
NUM_CLASSES = 50
EPOCHS = 12


def main():
    trainingExamples, trainingTargets, testingExamples, testingTargets = readDataFile(AUDIO_DATA)
    print "trainingExamples: ", trainingExamples.shape



main()
