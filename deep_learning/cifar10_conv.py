import numpy as np
from load_cifar10 import load_data
from models import Sequential
from layers import Dense, Dropout, Conv, Pool, Flatten, Activation, BatchNormalization
import optimizers
import objectives
import utils

rng = np.random.RandomState(1)
dataset = load_data()
x_train, y_train = dataset[0]
x_valid, y_valid = dataset[1]

shape = (3, 32, 32)
x_train = utils.reshape_img(x_train, shape)
x_valid = utils.reshape_img(x_valid, shape)

modelss = Sequential(shape, rng)

modelss.add(Conv(32, 3, 3))
modelss.add(BatchNormalization())
modelss.add(Activation('relu'))

modelss.add(Conv(32, 3, 3))
modelss.add(BatchNormalization())
modelss.add(Activation('relu'))
modelss.add(Pool())

modelss.add(Flatten())

modelss.add(Dense(500))
modelss.add(BatchNormalization())
modelss.add(Activation('relu'))


modelss.add(Dense(10))
modelss.add(Activation('softmax'))

opt = optimizers.Adam()
loss = objectives.MulticlassLogLoss()

modelss.compile(100, 50, opt, loss)
modelss.fit(x_train, y_train, x_valid, y_valid)

