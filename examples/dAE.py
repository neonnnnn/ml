import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation, Dropout, Decoder
from ml.deeplearning.optimizers import SGD
from ml.deeplearning.objectives import MeanSquaredError
from ml.deeplearning.models import Sequential
from ml import utils

rng = np.random.RandomState(1234)
dataset = load_mnist.load_data()
x_train, y_train = dataset[0]
x_valid, y_valid = dataset[1]
x_test, y_test = dataset[2]
imshape = (1, 28, 28)

opt = SGD(lr=0.01, momentum=0.9)
loss = MeanSquaredError()
clf = Sequential(28*28, loss=loss, rng=rng, opt=opt, batch_size=20, nb_epoch=200, iprint=True)

clf.add(Dense(500))
clf.add(Activation('relu'))
clf.add(Dropout(p=0.5))

clf.add(Decoder(clf.layers[0]))
clf.add(Activation('sigmoid'))


clf.fit(x_train, x_train)

utils.visualize(clf.layers[0].W.get_value().T, (28, 28), (10, 10), "denoising-autoencoder_W.png")

output = clf.predict(x_test)

utils.visualize(output, (28, 28), (10, 10), "denoising-autoencoder_output.png")