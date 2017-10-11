import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation, Dropout, Decoder
from ml.deeplearning.optimizers import SGD
from ml.deeplearning.objectives import SquaredError
from ml.deeplearning.models import Sequential
from ml.utils import BatchIterator, visualize


if __name__ == '__main__':
    rng = np.random.RandomState(1234)
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]
    X_test, y_test = dataset[2]

    train_batches = BatchIterator([X_train, X_train])
    valid_batches = BatchIterator([X_valid, X_valid])

    opt = SGD(lr=0.01, momentum=0.9)
    loss = SquaredError()
    clf = Sequential(28*28, rng)

    clf.add(Dropout(0.2))
    clf.add(Dense(500))
    clf.add(Activation('relu'))
    clf.add(Decoder(clf.layers[1]))
    clf.add(Activation('sigmoid'))
    clf.compile(train_loss=loss, opt=opt)

    clf.fit(train_batches, 100, valid_batches)

    visualize(clf.layers[0].W.get_value().T, (28, 28), (10, 10),
              'imgs/denoising-autoencoder_W.png')

    output = clf.predict(BatchIterator(X_test, 100))

    visualize(output, (28, 28), (10, 10), 'imgs/denoising-autoencoder_output.png')
