import numpy as np
import load_mnist
from ml.utils import BatchIterator
from ml.deeplearning.layers import Dense, Activation, Dropout
from ml.deeplearning.optimizers import SGD
from ml.deeplearning.objectives import MulticlassLogLoss, L2Regularization
from ml.deeplearning.models import Sequential
from ml.deeplearning.serializers import save

if __name__ == '__main__':
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]
    X_test, y_test = dataset[2]

    rng = np.random.RandomState(123)
    train_loss = [MulticlassLogLoss(), L2Regularization()]
    test_loss = MulticlassLogLoss()
    opt = SGD(0.01, 0.9)
    clf = Sequential(784, rng=rng)

    for i in range(1):
        clf.add(Dense(500))
        clf.add(Activation('relu'))
        clf.add(Dropout(0.5))

    clf.add(Dense(10))
    clf.add(Activation('softmax'))

    clf.compile(train_loss=train_loss, test_loss=test_loss, opt=opt)
    train_batches = BatchIterator([X_train, y_train], 128, True)
    valid_batches = BatchIterator([X_valid, y_valid], 128)
    test_batches = BatchIterator([X_test, y_test], 128)

    clf.fit(train_batches, 10, valid_batches)

    """
    # clf.fit(train_batches, 100, valid_batches, iprint=False) is equivalent to following code:
    for i in range(100):
        for train_batch in train_batches:
            clf.fit_on_batch(train_batch)

        valid_loss = clf.test(valid_batches)
    """

    accuracy = clf.accuracy(BatchIterator(X_test, 128), y_test)

    save(clf, 'mnist_mlp.h5')
    print('Accuracy:{0}'.format(accuracy))
