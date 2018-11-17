import sys
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Conv1D, MaxPooling1D

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
NUM_CLASSES = 50



trainingExamples, trainingTargets, testingExamples, testingTargets = readDataFile(AUDIO_DATA)

trainingExamples = trainingExamples.reshape(1500, 9, 15, 1)
testingExamples = testingExamples.reshape(500, 9, 15, 1)

trainingTargets = keras.utils.to_categorical(trainingTargets, NUM_CLASSES)
testingTargets = keras.utils.to_categorical(testingTargets, NUM_CLASSES)

print "TrainingExamples: ", trainingExamples.shape
print "TrainingTargets: ", trainingTargets.shape


model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', input_shape = (9, 15, 1)))
model.add(Flatten())
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.summary()

model.fit(trainingExamples, trainingTargets, batch_size=250, epochs=10, verbose=1)
#score = model.evaluate()
