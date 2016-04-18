import numpy as np
from load_mnist import load_data
import models
from layers import Dense, Activation, BatchNormalization
import optimizers
import objectives

rng = np.random.RandomState(1234)
dataset = load_data()
x_train, y_train = dataset[0]
x_valid, y_valid = dataset[1]
imshape = (1, 28, 28)

clf = models.Sequential(28*28, rng)
# Input layer
clf.add(Dense(1000))
clf.add(BatchNormalization())
clf.add(Activation('relu'))

clf.add(Dense(1000))
clf.add(BatchNormalization())
clf.add(Activation('relu'))

clf.add(Dense(10))
clf.add(Activation('softmax'))

opt = optimizers.SGD(lr=0.1, momentum=0.9)
loss = objectives.MulticlassLogLoss()
clf.compile(batch_size=100, nb_epoch=100, opt=opt, loss=loss)
clf.fit(x_train, y_train, x_valid, y_valid, valid_mode='error_rate')


