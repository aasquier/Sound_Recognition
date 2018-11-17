import sys
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import RMSprop
sys.path.insert(0, '../')
from extractFeatures import readDataFile


#Data files
AUDIO_DATA      = "../data/audioData.csv"
IMAGE_DATA      = "../data/imageData.csv"
AUDIO_DATA_NORM = "../data/audioDataNormalized.csv"
IMAGE_DATA_NORM = "../data/imageDataNormalized.csv"

#Constants
ROWS = 1
COLS = 135
N_CLASSES = 50

# Hyper Parameters
BATCH_SIZE = 256
EPOCHS = 100
KERNEL_SIZE = (3)

np.random.seed(10)

def main():
    loss, accuracy = classifyData(AUDIO_DATA)
    #loss, accuracy = trainModel(IMAGE_DATA)

    print "\nLoss: ", loss, "%"
    print "Accuracy: ", accuracy, "%"


def classifyData(dataFileName):
    X_train, y_train, X_test, y_test = readDataFile(dataFileName)

    #Reshape to satisfy keras Conv1D input_shape
    X_train = X_train.reshape(-1, ROWS, COLS)
    X_test = X_test.reshape(-1, ROWS, COLS)

    #Convert into binary representation
    Y_train = keras.utils.to_categorical(y_train, N_CLASSES)
    Y_test = keras.utils.to_categorical(y_test, N_CLASSES)

    loss, accuracy = trainModel(X_train, X_test, Y_train, Y_test)
    return loss, accuracy


def trainModel(X_train, X_test, Y_train, Y_test):
    model = Sequential()

    model.add(Conv1D(32, kernel_size=KERNEL_SIZE, padding='same', activation='relu', input_shape = (ROWS, COLS), data_format = 'channels_first'))
    model.add(Conv1D(64, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(N_CLASSES))
    model.add(Dropout(0.50))
    model.add(Dense(N_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(X_train, Y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                verbose=1,
                validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)
    return round(score[0] * 100, 3), round(score[1] * 100, 3)

main()
