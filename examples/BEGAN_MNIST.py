import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation
from ml.deeplearning.optimizers import Adam
from ml.deeplearning.models import Sequential
from ml.deeplearning.normalizations import BatchNormalization
from ml.utils import BatchIterator, saveimg
from BEGAN import BEGAN
from ml.deeplearning.initializations import normal


def train_began_mnist(X_train, X_valid, z_dim, n_hidden, opt_gen, opt_dis, activation, epoch, batch_size):

    rng = np.random.RandomState(1)

    generator = Sequential(z_dim, rng)
    generator.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    generator.add(Activation(activation))
    generator.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    generator.add(Activation(activation))
    generator.add(Dense(X_train.shape[1], init=normal(0, 0.001)))
    generator.add(Activation('tanh'))

    encoder = Sequential(X_train.shape[1], rng)
    encoder.add(Dense(n_hidden, init=normal(0, 0.001)))
    encoder.add(Activation(activation))
    encoder.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    encoder.add(Activation(activation))
    encoder.add(Dense(50, init=normal(0, 0.001)))

    decoder = Sequential(50, rng)
    encoder.add(Activation(activation))
    decoder.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    decoder.add(Activation(activation))
    decoder.add(Dense(X_train.shape[1], init=normal(0, 0.001)))
    decoder.add(Activation('tanh'))

    gan = BEGAN(rng, generator=generator, encoder=encoder, decoder=decoder)
    gan.compile(opt_gen=opt_gen, opt_dis=opt_dis)

    z_plot = np.random.standard_normal((100, z_dim)).astype(np.float32)
    z_batch = BatchIterator([z_plot], 100)
    train_batches = BatchIterator([X_train*2. - 1.], batch_size, shuffle=True)
    z = np.random.standard_normal((X_valid.shape[0], z_dim)).astype(np.float32)
    valid_batches = BatchIterator([X_valid*2. - 1., z], batch_size)
    # training
    for i in xrange(epoch/10):
        print('epoch:{0}'.format(i+1))
        gan.fit(train_batches, valid_batches=valid_batches, epoch=10, iprint=True)
        generation = 127.5 * (generator.predict(z_batch)+1.)
        saveimg(generation.reshape(100, 28, 28), (10, 10),
                'imgs/BEGAN/BEGAN_MNIST_epoch' + str((i + 1)*10) + '.png')
    # save(gan, 'began_mnist.h5')
    return gan


def main():
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_valid, _ = dataset[1]
    batch_size = 64
    z_dim = 50
    n_hidden = 500
    epoch = 200
    opt_gen = Adam(lr=1e-4, beta1=0.5)
    opt_dis = Adam(lr=1e-4, beta1=0.5)
    train_began_mnist(X_train, X_valid, z_dim, n_hidden, opt_gen, opt_dis, 'elu', epoch, batch_size)

if __name__ == '__main__':
    main()