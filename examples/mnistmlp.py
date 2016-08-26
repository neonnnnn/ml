import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation, Dropout
from ml.deeplearning.optimizers import SGD
from ml.deeplearning.objectives import MulticlassLogLoss, L2Regularization
from ml.deeplearning.models import Sequential


if __name__ == '__main__':
    dataset = load_mnist.load_data()
    x_train, y_train = dataset[0]
    x_test, y_test = dataset[1]

    rng = np.random.RandomState(123)
    loss = [MulticlassLogLoss()]
    opt = SGD(0.01, 0.9)
    clf = Sequential(784, loss=loss, rng=rng, opt=opt, batch_size=20, nb_epoch=100, iprint=True)

    clf.add(Dropout(0.2))
    for i in range(5):
        clf.add(Dense(500))
        clf.add(Activation("relu"))
        clf.add(Dropout(0.5))

    clf.add(Dense(10))
    clf.add(Activation('softmax'))

    clf.fit(x_train, y_train, x_test, y_test, valid_mode="error_rate")