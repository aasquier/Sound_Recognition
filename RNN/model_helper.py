import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU 
from keras.optimizers import RMSprop

def build_and_compile(h_units=0, inp_shape=(), n_classes=0, h_act='tanh',
                      rec_act='sigmoid', out_act='softmax', bias=True,
                      lss='categorical_crossentropy', opt='rmsprop',
                      met=['accuracy']):
    model = Sequential()
    model.add(LSTM(h_units, input_shape=inp_shape, activation=h_act, recurrent_activation=rec_act,
                   unit_forget_bias=bias))
    model.add(Dense(n_classes, activation=out_act))
    model.compile(loss=lss, optimizer=opt, metrics=met)
    return model