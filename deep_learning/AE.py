import numpy as np
from load_mnist import load_data
import models
from layers import Dense, Activation, Decoder
import optimizers
import objectives
import utils

rng = np.random.RandomState(1234)
dataset = load_data()
x_train, y_train = dataset[0]
x_valid, y_valid = dataset[1]
x_test, y_test = dataset[2]
imshape = (1, 28, 28)


clf = models.Sequential(28*28, rng)
clf.add(Dense(500))
clf.add(Activation('sigmoid'))
clf.add(Decoder(clf.layers[0]))
clf.add(Activation('sigmoid'))

opt = optimizers.SGD(lr=0.1, momentum=0)
loss = objectives.MeanSquaredError()
clf.compile(batch_size=20, nb_epoch=20, opt=opt, loss=loss)
clf.fit(x_train, x_train, x_valid, x_valid)

utils.visualize(clf.layers[0].W.get_value().T, (28, 28), 10, "autoencoder_W.png")

output = clf.predict(x_test)

utils.visualize(output, (28, 28), 10,  "autoencoder_output.png")


