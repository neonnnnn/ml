import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation
from ml.deeplearning.optimizers import Adam
from ml.deeplearning.models import Sequential
from ml.deeplearning.distributions import Gaussian, Bernoulli
from ml.deeplearning.normalizations import BatchNormalization
from ml.utils import BatchIterator, saveimg
from VAE import VAE
from IWAE import IWAE
from ml.deeplearning.initializations import normal
from ml.deeplearning.serializers import save


def train_vae_mnist(X_train, X_valid, z_dim, n_hidden, lr, activation,
                    epoch, batch_size):
    saveimg(X_valid[:batch_size].reshape(batch_size, 28, 28)*255.0, (10, 10),
            'imgs/VAE/MNIST_Valid.png')

    rng = np.random.RandomState(1)

    encoder = Sequential(X_train.shape[1], rng)
    encoder.add(Dense(n_hidden, init=normal(0, 0.001)))
    encoder.add(Activation(activation))
    #encoder.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    #encoder.add(Activation(activation))
    encoder = Gaussian(mean_layer=Dense(z_dim), logvar_layer=Dense(z_dim),
                       network=encoder, rng=rng)
    decoder = Sequential(z_dim, rng)
    decoder.add(Dense(n_hidden, init=normal(0, 0.001)))
    decoder.add(Activation(activation))
    #decoder.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    #decoder.add(Activation(activation))
    decoder = Bernoulli(mean_layer=Dense(X_train.shape[1]), network=decoder, rng=rng)

    vae = IWAE(rng, encoder=encoder, decoder=decoder)
    opt = Adam(lr)
    vae.compile(opt=opt)

    z_plot = np.random.standard_normal((batch_size, z_dim)).astype(np.float32)
    z_plot[-1] = -z_plot[0]

    for i in range(batch_size):
        z_plot[i] = ((batch_size-i)*z_plot[0] + i*z_plot[99])/batch_size

    def binarize(x):
        return rng.binomial(1, x).astype(np.float32)

    train_batches = BatchIterator([X_train], batch_size)
    z_batch = BatchIterator(z_plot, batch_size)
    X_valid_bacth = BatchIterator(X_valid[:batch_size], batch_size, aug=binarize)

    # training
    for i in xrange(epoch/10):
        print('epoch:{0}'.format(i+1))
        vae.fit(train_batches, epoch=10, iprint=True)
        reconstract = vae.predict(X_valid_bacth)
        print(np.sum((X_valid[:100] - reconstract)**2)/100.)
        generation = decoder.predict(z_batch)
        saveimg(255.*generation.reshape(100, 28, 28), (10, 10),
                'imgs/VAE/IWAE_MNIST_epoch' + str((i+1)*10) + '.png')
        saveimg(255.*reconstract.reshape(100, 28, 28), (10, 10),
                'imgs/VAE/IWAE_MNIST_reconstruct_epoch' + str((i+1)*10) + '.png')
    return vae


def main():
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]
    batch_size = 100
    z_dim = 50
    n_hidden = 600
    epoch = 200
    train_vae_mnist(np.vstack((X_train, X_valid)), X_valid, z_dim, n_hidden, 1e-3, 'relu', epoch, batch_size)

if __name__ == '__main__':
    main()

