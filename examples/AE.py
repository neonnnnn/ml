import numpy as np
from load_mnist import load_data
from ml.deeplearning.layers import Dense, Activation, Dropout, Decoder
from ml.deeplearning.optimizers import SGD, AdaGrad
from ml.deeplearning.objectives import SquaredError
from ml.deeplearning.models import Sequential
from ml import utils
if __name__ == '__main__':
    rng = np.random.RandomState(1234)
    dataset = load_data()
    x_train, y_train = dataset[0]
    x_valid, y_valid = dataset[1]
    x_test, y_test = dataset[2]
    imshape = (1, 28, 28)

    opt = SGD(lr=0.01, momentum=0.9)
    loss = SquaredError()

    clf = Sequential(28*28, rng)

    clf.add(Dense(500))
    clf.add(Activation('relu'))
    clf.add(Decoder(clf.layers[0]))
    clf.add(Activation('sigmoid'))
    clf.compile(opt=opt, loss=loss)

    clf.fit(x_train, x_train)

    W = clf.layers[0].W.get_value().T.reshape(500, 28, 28)
    utils.visualize(W, (10, 10), "imgs/autoencoder_W.png")

    output = clf.predict(x_test).reshape(x_test.shape[0], 28, 28)

    utils.visualize(output, (10, 10),  "imgs/autoencoder_output.png")
