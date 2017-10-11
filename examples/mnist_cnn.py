import numpy as np
import load_mnist
from ml.deeplearning.layers import (Dense, Activation, ConvCUDNN,
                                    Pool, Flatten)
from ml.deeplearning.normalizations import BatchNormalization
from ml.deeplearning.optimizers import SGD
from ml.deeplearning.objectives import MulticlassLogLoss
from ml.deeplearning.models import Sequential
from ml.utils import BatchIterator

if __name__ == '__main__':
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_test, y_test = dataset[1]
    X_train = X_train.reshape(50000, 1, 28, 28)
    X_test = X_test.reshape(10000, 1,  28, 28)
    train_batches = BatchIterator([X_train, y_train], 128, True)
    valid_batches = BatchIterator([X_test, y_test], 128, False)

    rng = np.random.RandomState(123)
    loss = [MulticlassLogLoss()]
    opt = SGD(0.01, 0.9)
    clf = Sequential((1, 28, 28), rng=rng)

    clf.add(BatchNormalization(ConvCUDNN(32, 5, 5)))
    clf.add(Activation('relu'))
    clf.add(Pool())

    clf.add(BatchNormalization(ConvCUDNN(32, 5, 5)))
    clf.add(Activation('relu'))
    clf.add(Pool())

    clf.add(Flatten())

    clf.add(Dense(500))
    clf.add(Activation('relu'))

    clf.add(Dense(10))
    clf.add(Activation('softmax'))

    clf.compile(opt=opt, train_loss=loss)

    clf.fit(train_batches, 100, valid_batches)
    """
    # clf.fit(train_batches, 100, valid_batches, iprint=False) is equivalent
    # to following way:
    for i in range(100):
        for train_batch in train_batches:
            clf.fit_on_batch(train_batch)

        valid_loss = clf.test(valid_batches)
    """

    accuracy = clf.accuracy(BatchIterator(X_test, 128), y_test)
    print('Accuracy:{0}'.format(accuracy))
