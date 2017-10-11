import theano.tensor as T
import theano
import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, BatchNormalization, Activation
from ml.deeplearning.optimizers import Adam
from ml.deeplearning.models import Model, Sequential
from ml.deeplearning.distributions import Distribution
from ml.deeplearning.distributions import Gaussian, Bernoulli
from ml.utils import BatchIterator, onehot, saveimg
from CVAE import CVAE
from VAE_MNIST import binarize
from ml.deeplearning.theanoutils import variable


class Encoder(Gaussian):
    def forward(self, x, y=None, sampling=True, train=False):
        if y is None:
            inputs = x
        else:
            inputs = T.concatenate([x, y], axis=1)
        output = super(Encoder, self).forward(inputs,
                                              sampling=sampling,
                                              train=train)
        return output

    def function(self, x, y, mode='pred'):
        function = super(Encoder, self).function(T.concatenate([x, y], axis=1), mode=mode)
        return function


class Decoder(Bernoulli):
    def forward(self, z, y=None, sampling=True, train=False):
        if y is None:
            inputs = z
        else:
            inputs = T.concatenate([z, y], axis=1)
        mean = super(Decoder, self).forward(inputs, sampling=sampling, train=train)
        return mean

    def function(self, z, y,  mode='pred'):
        function = super(Decoder, self).function(T.concatenate([z, y], axis=1), mode=mode)
        return function


def train_cvae_mnist():
    # load data
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]
    y_train = onehot(y_train).astype(np.float32)
    y_valid = onehot(y_valid).astype(np.float32)
    y = onehot((np.arange(100)/10) % 10).astype(np.float32)
    # set params
    batch_size = 100
    epoch = 200
    rng1 = np.random.RandomState(1)
    z_dim = 50

    # make encoder, decoder
    encoder = Sequential(n_in=X_train.shape[1]+y_train.shape[1], rng=rng1)
    encoder.add(Dense(600))
    encoder.add(BatchNormalization())
    encoder.add(Activation('softplus'))
    encoder.add(Dense(600))
    encoder.add(BatchNormalization())
    encoder.add(Activation('softplus'))
    encoder = Encoder(network=encoder, mean_layer=Dense(z_dim), logvar_layer=Dense(z_dim), rng=rng1)

    decoder = Sequential(n_in=z_dim+y_train.shape[1], rng=rng1)
    decoder.add(BatchNormalization())
    decoder.add(Activation('softplus'))
    decoder.add(Dense(600))
    decoder.add(BatchNormalization())
    decoder.add(Activation('softplus'))
    decoder = Decoder(rng=rng1, network=decoder, mean_layer=Dense(X_train.shape[1]))

    # concat encoder and decoder, and define loss
    cvae = CVAE(rng1, encoder=encoder, decoder=decoder)
    opt = Adam(lr=3e-4)
    cvae.compile(opt=opt, train_loss=None)

    f_encode = encoder.function(variable(X_train), variable(y), mode='pred')
    f_decode = decoder.function(variable(X_train), variable(y), mode='pred')

    train_batches = BatchIterator([X_train, y_train], batch_size, aug=[binarize, None])
    # train
    for i in xrange(epoch/10):
        print('epoch:{0}-{1}'.format(i*10, (i+1)*10))
        cvae.fit(train_batches, epoch=10)
        z = f_encode(np.concatenate((X_valid[:100], y_valid[:100]), axis=1))[0]
        z = np.tile(z[:10, :z_dim], (10, 1)).reshape(100, z_dim)
        reconstract = f_decode(np.concatenate((z, y), axis=1))[0]
        plot = 255 * np.vstack((X_valid[:10], reconstract))
        saveimg(plot.reshape(110, 28, 28).astype(np.uint8), (11, 10),
                'imgs/CVAE/CVAE_MNIST_analogy_epoch' + str((i+1)*10) + '.png')
        del plot
        del z

if __name__ == '__main__':
    train_cvae_mnist()
