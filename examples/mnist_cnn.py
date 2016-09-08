import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation, Dropout, Conv, Pool, Flatten
from ml.deeplearning.optimizers import SGD
from ml.deeplearning.objectives import MulticlassLogLoss, L2Regularization
from ml.deeplearning.models import Sequential


if __name__ == '__main__':
    dataset = load_mnist.load_data()
    x_train, y_train = dataset[0]
    x_test, y_test = dataset[1]
    x_train = x_train.reshape(50000, 1, 28, 28)
    x_test = x_test.reshape(10000, 1,  28, 28)
    rng = np.random.RandomState(123)
    loss = [MulticlassLogLoss()]
    opt = SGD(0.01, 0.9)
    clf = Sequential((1, 28, 28), rng=rng)

    clf.add(Conv(32, 5, 5))
    clf.add(Activation("relu"))
    clf.add(Pool())

    clf.add(Conv(32, 5, 5))
    clf.add(Activation("relu"))
    clf.add(Pool())

    clf.add(Flatten())

    clf.add(Dense(500))
    clf.add(Activation("relu"))
    clf.add(Dense(10))
    clf.add(Activation('softmax'))

    clf.compile(opt=opt, loss=loss)
    clf.fit(x_train, y_train, x_test, y_test, valid_mode="loss")