import sys
sys.path.insert(0, '../')
from extractFeatures import readDataFile

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import RMSprop



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


def main():
    aud_X_train, aud_y_train, aud_X_test, aud_y_test = readDataFile(AUDIO_DATA)
    im_X_train, im_y_train, im_X_test, im_y_test = readDataFile(IMAGE_DATA)


def trainModel(dataFileName):


# aud_X_train = aud_X_train.reshape(aud_X_train.shape[0], ROWS, COLS)
aud_X_train = aud_X_train.reshape(-1, ROWS, COLS)
aud_X_test = aud_X_test.reshape(-1, ROWS, COLS)
aud_X_train.shape

im_X_train = im_X_train.reshape(-1, ROWS, COLS)
im_X_test = im_X_test.reshape(-1, ROWS, COLS)

aud_Y_train = keras.utils.to_categorical(aud_y_train, N_CLASSES)
aud_Y_test = keras.utils.to_categorical(aud_y_test, N_CLASSES)
im_Y_train = keras.utils.to_categorical(im_y_train, N_CLASSES)
im_Y_test = keras.utils.to_categorical(im_y_test, N_CLASSES)

# RNN LSTM
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

model.fit(aud_X_train, aud_Y_train,
         batch_size=BATCH_SIZE,
         epochs=EPOCHS,
         verbose=1,
         validation_data=(aud_X_test, aud_Y_test))

score = model.evaluate(aud_X_test, aud_Y_test, verbose=0)


# model.fit(im_X_train, im_Y_train,
#          batch_size=BATCH_SIZE,
#          epochs=EPOCHS,
#          verbose=1,
#          validation_data=(im_X_test, im_Y_test))
#
# score = model.evaluate(im_X_test, im_Y_test, verbose=0)

print "Loss: ", round(score[0], 3) * 100, "%"
print "Accuracy: ", round(score[1], 3) * 100, "%"

main()
