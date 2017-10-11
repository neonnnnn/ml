import numpy as np
from load_mnist import load_data
from ml.deeplearning.layers import Dense, Activation, Decoder
from ml.deeplearning.optimizers import SGD
from ml.deeplearning.objectives import SquaredError
from ml.deeplearning.models import Sequential
from ml.utils import visualize, BatchIterator


def autoencoder(n_in, n_hidden=500, rng=np.random.RandomState(1234),
                activations=['relu', 'sigmoid']):
    clf = Sequential(n_in, rng)
    clf.add(Dense(n_hidden))
    clf.add(Activation(activations[0]))
    clf.add(Decoder(clf.layers[0]))
    clf.add(Activation(activations[1]))
    
    return clf

if __name__ == '__main__':
    dataset = load_data()
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]
    X_test, y_test = dataset[2]

    opt = SGD(lr=0.01, momentum=0.9)
    loss = SquaredError()

    clf = autoencoder(28*28)
    clf.compile(opt=opt, train_loss=loss)
    train_batches = BatchIterator([X_train, X_train], batch_size=128,
                                  shuffle=True)
    valid_batches = BatchIterator([X_valid, X_valid], batch_size=128,
                                  shuffle=False)
    clf.fit(train_batches, 100, valid_batches)

    W = clf.layers[0].W.get_value().T.reshape(500, 28, 28)
    visualize(W, (10, 10), 'imgs/autoencoder_W.png')

    output = clf.predict(X_test).reshape(X_test.shape[0], 28, 28)

    visualize(output, (10, 10),  'imgs/autoencoder_output.png')
