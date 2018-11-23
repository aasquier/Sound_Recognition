
import sys
sys.path.insert(0, '/Users/carsoncook/Dev/CS445/Group_Project_cs445')
import extractFeatures as ef

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU 
from keras.optimizers import RMSprop

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Constants
FEATURES = 135
ROWS = 3
COLS = 45
N_CLASSES = 10

# Hyper Parameters
UNITS = 128
BATCH_SIZE = 256
EPOCHS = 10
# DROPOUT = (.25, .5)
IMAGE_PATH = 'images/'
ACC_PATH = 'accuracies/'

np.random.seed(10)

# RNN LSTM
def build_and_run_model(X_train, X_test, Y_train, Y_test, units, rows, cols, batch_size, epochs, fold, prefix):
    X_train = X_train.reshape(-1, rows, cols)
    X_test = X_test.reshape(-1, rows, cols)
    
    model = Sequential()
    model.add(LSTM(units, input_shape=(rows, cols),
                   return_sequences=True,
                   activation='tanh',
                   recurrent_activation ='sigmoid',
                   unit_forget_bias=True,
                  ))
    model.add(LSTM(units, input_shape=(rows, cols),
                   activation='tanh',
                   recurrent_activation ='sigmoid',
                   unit_forget_bias=True,
                  ))
    model.add(Dense(N_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X_train, Y_train,
             batch_size=batch_size,
             epochs=epochs,
             verbose=1,
             validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    preds = model.predict_classes(X_test)
    y_test = np.argmax(Y_test, axis=1)
    # cm = confusion_matrix(y_test, preds)
    output = np.vstack((preds, y_test)).T
    file_name = '{}{}_fold_{}_accs.csv'.format(ACC_PATH, prefix, fold)
    np.savetxt(file_name, output, fmt='%d', delimiter=',', header='predicted,actual')
    print("Score: ", score)
    return score[1]


# Assess model using cross validation
def run_k_fold_cv(k, acc_file, units, rows, cols):
    accs = np.zeros(shape=(k,3), dtype='float64')
    for i in range(k):
        (aud_X_train, aud_X_test, aud_y_train, aud_y_test,
        im_X_train, im_X_test, im_y_train, im_y_test,
        aud_X_train_norm, aud_X_test_norm, aud_y_train_norm, aud_y_test_norm,
        im_X_train_norm, im_X_test_norm, im_y_train_norm, im_y_test_norm) = ef.crossValidationIteration(i)  
        
        # Smash features together
        full_X_train = np.concatenate((aud_X_train, im_X_train))
        full_X_test = np.concatenate((aud_X_test, im_X_test))
        full_y_train = np.hstack((aud_y_train, im_y_train))
        full_y_test = np.hstack((aud_y_test, im_y_test))

        # One Hot Encoding
        aud_Y_train = keras.utils.to_categorical(aud_y_train, N_CLASSES)
        aud_Y_test = keras.utils.to_categorical(aud_y_test, N_CLASSES)
        im_Y_train = keras.utils.to_categorical(im_y_train, N_CLASSES)
        im_Y_test = keras.utils.to_categorical(im_y_test, N_CLASSES)
        full_Y_train = keras.utils.to_categorical(full_y_train, N_CLASSES)
        full_Y_test = keras.utils.to_categorical(full_y_test, N_CLASSES)
        
        accs[i, 0] = build_and_run_model(aud_X_train, aud_X_test, aud_Y_train, aud_Y_test,
                                    units, rows, cols, BATCH_SIZE, EPOCHS, i, 'aud')
        accs[i, 1] = build_and_run_model(im_X_train, im_X_test, im_Y_train, im_Y_test,
                                    units, rows, cols, BATCH_SIZE, EPOCHS, i, 'im')
        accs[i, 2] = build_and_run_model(full_X_train, full_X_test, full_Y_train, full_Y_test,
                                    units, rows, cols, BATCH_SIZE, EPOCHS, i, 'comb')

        with open (acc_file, 'a') as outfile:
            outfile.write('\n[Fold: {}]\nAudio Acc: {}\nImage Acc: {}\nCombined Acc: {}'.format(i, accs[i, 0], accs[i, 1], accs[i, 2]))
    with open (acc_file, 'a') as outfile:
        avgs = accs.mean(axis = 0)
        outfile.write('\nAvg. Audio Acc: {}, Avg. Image Acc: {}, Avg. Combined Acc: {}'.format(
            avgs[0], avgs[1], avgs[2]
        ))


run_k_fold_cv(10, 'accuracies_v2.txt', UNITS, ROWS, COLS)

