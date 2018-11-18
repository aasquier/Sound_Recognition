import sys
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import RMSprop
sys.path.insert(0, '../')
from extractFeatures import readDataFile, crossValidationIteration


#Data files
AUDIO_DATA      = "../data/audioData.csv"
IMAGE_DATA      = "../data/imageData.csv"
AUDIO_DATA_NORM = "../data/audioDataNormalized.csv"
IMAGE_DATA_NORM = "../data/imageDataNormalized.csv"

#Constants
ROWS = 1
COLS = 135
N_CLASSES = 50
K_ITERATIONS = 10

# Hyper Parameters
BATCH_SIZE = 256
EPOCHS = 100
KERNEL_SIZE = (3)

np.random.seed(10)

''' Starting place for this program '''
def main():
    #classifyData(AUDIO_DATA)
    #classifyData(IMAGE_DATA)
    classifyDataWithCrossValidation()


''' Classifies a CNN with the given data set '''
def classifyData(dataFileName):
    X_train, y_train, X_test, y_test = readDataFile(dataFileName)

    #Reshape to satisfy keras Conv1D input_shape
    X_train = X_train.reshape(-1, ROWS, COLS)
    X_test = X_test.reshape(-1, ROWS, COLS)

    #Convert into binary representation
    Y_train = keras.utils.to_categorical(y_train, N_CLASSES)
    Y_test = keras.utils.to_categorical(y_test, N_CLASSES)

    loss, accuracy = trainModel(X_train, X_test, Y_train, Y_test)

    print "Accuracy: ", accuracy, "%"
    print "Loss: ", loss, "%"


''' Classifies a CNN with Cross Validation '''
def classifyDataWithCrossValidation():
    accuracyTotals = np.zeros((K_ITERATIONS, 3))
    for i in range(K_ITERATIONS):
        audioTrainingExamples, audioTestingExamples, audioTrainingTargets, audioTestingTargets, videoTrainingExamples, videoTestingExamples, videoTrainingTargets, videoTestingTargets, audioTrainingExamplesNormalized, audioTestingExamplesNormalized, audioTrainingTargetsNormalized, audioTestingTargetsNormalized, videoTrainingExamplesNormalized, videoTestingExamplesNormalized, videoTrainingTargetsNormalized, videoTestingTargetsNormalized = crossValidationIteration(i)

        fullTrainingExamples = np.concatenate((audioTrainingExamples, videoTrainingExamples))
        fullTestingExamples  = np.concatenate((audioTestingExamples, videoTestingExamples))
        fullTrainingTargets  = np.hstack((audioTrainingTargets, videoTrainingTargets))
        fullTestingTargets   = np.hstack((audioTestingTargets, videoTestingTargets))

        fullTrainingExamples = fullTrainingExamples.reshape(-1, ROWS, COLS)
        fullTestingExamples = fullTestingExamples.reshape(-1, ROWS, COLS)
        audioTrainingExamples = audioTrainingExamples.reshape(-1, ROWS, COLS)
        audioTestingExamples = audioTestingExamples.reshape(-1, ROWS, COLS)
        videoTrainingExamples = videoTrainingExamples.reshape(-1, ROWS, COLS)
        videoTestingExamples = videoTestingExamples.reshape(-1, ROWS, COLS)

        fullTrainingTargets = keras.utils.to_categorical(fullTrainingTargets, N_CLASSES)
        fullTestingTargets = keras.utils.to_categorical(fullTestingTargets, N_CLASSES)
        audioTrainingTargets = keras.utils.to_categorical(audioTrainingTargets, N_CLASSES)
        audioTestingTargets = keras.utils.to_categorical(audioTestingTargets, N_CLASSES)
        videoTrainingTargets = keras.utils.to_categorical(videoTrainingTargets, N_CLASSES)
        videoTestingTargets = keras.utils.to_categorical(videoTestingTargets, N_CLASSES)

        _, audioAccuracy = trainModel(audioTrainingExamples, audioTestingExamples, audioTrainingTargets, audioTestingTargets)
        _, videoAccuracy = trainModel(videoTrainingExamples, videoTestingExamples, videoTrainingTargets, videoTestingTargets)
        _, fullAccuracy = trainModel(fullTrainingExamples, fullTestingExamples, fullTrainingTargets, fullTestingTargets)

        accuracyTotals[i, 0] = audioAccuracy
        accuracyTotals[i, 1] = videoAccuracy
        accuracyTotals[i, 2] = fullAccuracy

        print "\nAudio Accuracy [", i + 1, "]:  ", audioAccuracy
        print "Image Accuracy [", i + 1, "]:    ", videoAccuracy
        print "Combined Accuracy [", i + 1, "]: ", fullAccuracy


    accuracyTotals = (np.sum(accuracyTotals, axis=0)) / K_ITERATIONS

    print "\nAverages for ", K_ITERATIONS, " iterations:"
    print "Average Audio Accuracy [", i + 1, "]:    ", round(accuracyTotals[0], 2), "%"
    print "Average Image Accuracy [", i + 1, "]:    ", round(accuracyTotals[1], 2), "%"
    print "Average Combined Accuracy [", i + 1, "]: ", round(accuracyTotals[2], 2), "%"


''' Trains a Convolutional Neural Network and returns the accuracy and loss '''
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
    return round(score[0] * 100, 2), round(score[1] * 100, 2)


main()
