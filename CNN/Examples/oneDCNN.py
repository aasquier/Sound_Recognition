### LOAD PACKAGES
from numpy.random import seed
from pandas import read_csv, DataFrame
from sklearn.preprocessing import minmax_scale
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten

### PREPARE THE DATA
df = read_csv("credit_count.txt")
Y = df[df.CARDHLDR == 1].DEFAULT
X = minmax_scale(df[df.CARDHLDR == 1].ix[:, 2:12], axis = 0)
y_train = Y.values
x_train = X.reshape(X.shape[0], X.shape[1], 1)

### FIT A 1D CONVOLUTIONAL NEURAL NETWORK
seed(2017)
conv = Sequential()
conv.add(Conv1D(20, 4, input_shape = x_train.shape[1:3], activation = 'relu'))
conv.add(MaxPooling1D(2))
conv.add(Flatten())
conv.add(Dense(1, activation = 'sigmoid'))
sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
conv.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
conv.fit(x_train, y_train, batch_size = 500, epochs = 100, verbose = 0)
