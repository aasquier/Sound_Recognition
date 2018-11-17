import sys
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D

sys.path.insert(0, '../')
from extractFeatures import readDataFile

np.set_printoptions(threshold=np.nan) #Configure numpy's array printing to show everything

#Data files
AUDIO_DATA      = "../data/audioData.csv"
IMAGE_DATA      = "../data/imageData.csv"
AUDIO_DATA_NORM = "../data/audioDataNormalized.csv"
IMAGE_DATA_NORM = "../data/imageDataNormalized.csv"

#Constants
TRAINING_M = 1500
TESTING_M = 2000 - TRAINING_M
FEATURES = 135
INPUT_SHAPE = (TRAINING_M, FEATURES) #(BATCH_SIZE, FEATURES, 1)

#Hyper Parameters
BATCH_SIZE = 100
NUM_CLASSES = 50
NUM_FILTERS = 32
NUM_DENSE_LAYERS = NUM_FILTERS * 2 #Used to predict the labels
KERNEL_SIZE = (3)
POOL_SIZE = (2)                 #Used for downsampling
EPOCHS = 12
DROP_OUT = (0.25, 0.50)            #Reduces overfitting


def main():
    trainingExamples, trainingTargets, testingExamples, testingTargets = setupCNN()
    testLoss, testAccuracy = trainCNN(trainingExamples, trainingTargets, testingExamples, testingTargets)
    print('Test loss:', testLoss)
    print('Test accuracy:', testAccuracy)

def trainCNN(trainingExamples, trainingTargets, testingExamples, testingTargets):
    model = Sequential()
    #params: Conv2D(kernel size (dimensions of the weights), input_shape = (width, height, channels)) #Where channels is like RGB - I think ours is just 1 (maybe 0)??
    model.add(Conv1D(NUM_FILTERS, kernel_size=KERNEL_SIZE,
                     activation='relu',
                     input_shape=INPUT_SHAPE))
    model.add(Conv1D(NUM_FILTERS, kernel_size=KERNEL_SIZE,
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))
    model.add(Dropout(DROP_OUT[0]))
    model.add(Flatten())
    model.add(Dense(NUM_DENSE_LAYERS, activation='relu'))
    model.add(Dropout(DROP_OUT[1]))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    model.fit(trainingExamples, trainingTargets,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(testingExamples, testingTargets))
            # validation_data=None)

    score = model.evaluate(testingExamples, testingTargets, verbose=0)
    return score[0], score[1]



def setupCNN():
    trainingExamples, trainingTargets, testingExamples, testingTargets = readDataFile(AUDIO_DATA)

    trainingExamples = trainingExamples.reshape(TRAINING_M, FEATURES, 1)
    testingExamples = testingExamples.reshape(TESTING_M, FEATURES, 1)

    # convert class vectors to binary class matrices
    trainingTargets = keras.utils.to_categorical(trainingTargets, NUM_CLASSES)
    testingTargets = keras.utils.to_categorical(testingTargets, NUM_CLASSES)
    return trainingExamples, trainingTargets, testingExamples, testingTargets



main()
