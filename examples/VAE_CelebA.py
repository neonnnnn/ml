import theano.tensor as T
import theano
import numpy as np
import os
import timeit
from scipy.io import loadmat
from ml.deeplearning.layers import (Dense, BatchNormalization, Flatten,
                                    Reshape, DeconvCUDNN, ConvCUDNN,
                                    Activation)
from ml.deeplearning.activations import leakyrelu
from ml.deeplearning.optimizers import Adam
from ml.deeplearning.models import Model, Sequential
from ml.deeplearning.objectives import Regularization
from ml import utils
import math
import pickle


class KLD(Regularization):
    def __init__(self, z_dim, mode=1):
        super(KLD, self).__init__(weight=1.0)
        self.z_dim = z_dim
        self.mode = mode

    def calc(self, mean, logvar):
        output = -T.sum(0.5+logvar/2.-T.exp(logvar)/2.-(mean*mean)/2., axis=1)
        return output


class CVAE(Model):
    def __init__(self, rng, encoder, decoder, z_dim):
        self.z_dim = z_dim
        super(CVAE, self).__init__(rng, encoder=encoder, decoder=decoder)

    def forward(self, x, train):
        srng = T.shared_randomstreams.RandomStreams(self.rng.randint(999))
        e_mean, e_logvar = self.encoder.forward(x, train)
        epsilon = srng.normal(size=(e_mean.shape[0], self.z_dim), avg=0.,
                              std=1., dtype=e_mean.dtype)
        z = e_mean + epsilon * T.exp(0.5 * e_logvar)
        d_mean, d_logvar = self.decoder.forward(z, train)

        return e_mean, e_logvar, d_mean, d_logvar

    def function(self, x, w, opt=None, train=True):
        e_mean, e_logvar, d_mean, d_logvar = self.forward(x, train)

        if train:
            kld = KLD(self.z_dim, mode=0)
            x_diff = x - d_mean
            recons = T.sum((x_diff*x_diff)/(2.*T.exp(d_logvar)) + d_logvar/2.
                           + math.log(2*math.pi)/2., axis=(1, 2, 3))
            # cost for encoder
            cost = T.mean(recons + kld.calc(e_mean, e_logvar))
            updates = opt.get_updates(cost, self.params) + self.get_updates()
            function = theano.function(inputs=[x],
                                       outputs=[cost], updates=updates)
        else:
            function = theano.function(inputs=[x], outputs=[d_mean])

        return function


class Encoder(Model):
    def __init__(self, rng, encoder_x, encoder_mean, encoder_logvar, z_dim):
        self.z_dim = z_dim
        super(Encoder, self).__init__(rng, encoder_x=encoder_x,
                                      encoder_mean=encoder_mean,
                                      encoder_logvar=encoder_logvar)

    def forward(self, x, train):
        output = self.encoder_x(x, train)
        mean = self.encoder_mean(output, train)
        logvar = self.encoder_logvar(output, train)
        return mean, logvar

    def function(self, train):
        x = T.tensor4('x')
        mean, logvar = self.forward(x, False)
        return theano.function(inputs=[x], outputs=[mean])


class Decoder(Model):
    def __init__(self, rng, decoder_z, decoder_mean,
                 decoder_logvar):
        super(Decoder, self).__init__(rng, decoder_z=decoder_z,
                                      decoder_mean=decoder_mean,
                                      decoder_logvar=decoder_logvar)

    def forward(self, z, train=False):
        h5 = self.decoder_z(z, train)
        mean = self.decoder_mean(h5, train)
        logvar = self.decoder_logvar(h5, train)
        return mean, logvar

    def function(self, train, discriminator=None):
        z = T.matrix('z')
        mean, logvar = self.forward(z, train=False)
        return theano.function(inputs=[z], outputs=[mean])


def x_encoder(rng):
    clf = Sequential((3, 64, 64), rng=rng, iprint=False)
    clf.add(ConvCUDNN(64, 5, 5, init='normal', subsample=(2, 2),
                      border_mode=(2, 2)))
    clf.add(BatchNormalization(moving=True, trainable=False))
    clf.add(Activation('leakyrelu'))

    clf.add(ConvCUDNN(128, 5, 5, init='normal', subsample=(2, 2),
                      border_mode=(2, 2)))
    clf.add(BatchNormalization(moving=True, trainable=False))
    clf.add(Activation('leakyrelu'))

    clf.add(ConvCUDNN(256, 5, 5, init='normal', subsample=(2, 2),
                      border_mode=(2, 2)))
    clf.add(BatchNormalization(moving=True, trainable=False))
    clf.add(Activation('leakyrelu'))

    clf.add(ConvCUDNN(256, 5, 5, init='normal', subsample=(2, 2),
                      border_mode=(2, 2)))
    clf.add(BatchNormalization(moving=True, trainable=False))
    clf.add(Activation('leakyrelu'))

    clf.add(Flatten())
    clf.add(Dense(1024))
    clf.add(BatchNormalization(moving=True, trainable=False))
    clf.add(Activation('leakyrelu'))

    return clf


def z_decoder(rng, z_dim):
    g = Sequential(z_dim, rng=rng, iprint=False)
    g.add(Dense(256*4*4, init='normal'))
    g.add(Activation('relu'))
    g.add(Reshape((256, 4, 4)))

    g.add(DeconvCUDNN(256, 5, 5, (256, 8, 8), init='normal',
                      subsample=(2, 2), border_mode=(2, 2)))
    g.add(BatchNormalization(moving=True, trainable=False))
    g.add(Activation('leakyrelu'))

    g.add(DeconvCUDNN(128, 5, 5, (128, 16, 16), init='normal',
                      subsample=(2, 2), border_mode=(2, 2)))
    g.add(BatchNormalization(moving=True, trainable=False))
    g.add(Activation('leakyrelu'))

    g.add(DeconvCUDNN(64, 5, 5, (64, 32, 32), init='normal',
                      subsample=(2, 2), border_mode=(2, 2)))
    g.add(BatchNormalization(moving=True, trainable=False))
    g.add(Activation('leakyrelu'))

    return g


def train_cvae_celeba():
    # set params
    batch_size = 100
    sqrtbs = int(batch_size ** 0.5)
    epoch = 200
    rng = np.random.RandomState(1)

    z_dim = 128
    # load data
    filedir = os.path.expanduser('~') + '/dataset/celeba_normalize_mat/'
    fs = np.asarray(os.listdir(filedir))
    n_data = len(fs)
    print n_data
    train_fs = fs[:-batch_size]
    valid_fs = fs[-batch_size:]
    X_train = np.zeros((batch_size, 3, 64, 64), dtype=np.float32)
    X_valid = np.zeros((batch_size, 3, 64, 64), dtype=np.float32)
    params_dir = '/media/neon/HDD/params/'
    for i, valid_f in enumerate(valid_fs):
        data = loadmat(filedir + valid_f)
        X_valid[i] = data['img']

    # make encoder, decoder, discriminator, and cvaegan
    encoder_x = x_encoder(rng)
    encoder_mean = Dense(n_in=1024, n_out=z_dim, init='normal')
    encoder_logvar = Dense(n_in=1024, n_out=z_dim, init='zero')
    encoder = Encoder(rng, encoder_x=encoder_x, encoder_mean=encoder_mean,
                      encoder_logvar=encoder_logvar, z_dim=z_dim)
    decoder_mean = DeconvCUDNN(3, 5, 5, n_out=(3, 64, 64), n_in=(64, 32, 32),
                               init='normal', subsample=(2, 2),
                               border_mode=(2, 2))
    decoder_logvar = DeconvCUDNN(3, 5, 5, n_out=(3, 64, 64), n_in=(64, 32, 32),
                                 init='normal', subsample=(2, 2),
                                 border_mode=(2, 2))
    decoder_z = z_decoder(rng, z_dim)
    decoder = Decoder(rng, decoder_z=decoder_z,
                      decoder_mean=decoder_mean, decoder_logvar=decoder_logvar)
    cvae = CVAE(rng, encoder=encoder, decoder=decoder, z_dim=z_dim)
    opt = Adam(lr=3e-4, beta1=0.5)

    f_train = cvae.function(opt, True)
    f_encode = encoder.function(train=False)
    f_decode = decoder.function(train=False)

    n_batches = (n_data - batch_size) // batch_size
    # train
    for i in xrange(epoch):
        start = 0
        print('epoch:{0}'.format(i + 1))
        s = timeit.default_timer()
        cost = 0
        for j in xrange(10):
            idx = np.random.randint(n_data - batch_size, size=batch_size)
            for k, train_f in enumerate(train_fs[list(idx)]):
                data = loadmat(filedir + train_f)
                X_train[k] = data['img']
            cost += f_train(X_train)[0] / n_batches
            start += batch_size
            e = timeit.default_timer()
            utils.progbar(j + 2, n_data // batch_size, e - s)
        print(' cost:{0}'.format(batch_size*cost/(n_data-batch_size)))

        if (i + 1) % 1 == 0:
            z = f_encode(X_valid)[0]
            analogy = (f_decode(z)[0] + 1) / 2
            analogy[analogy > 1] = 1.
            analogy[analogy < 0] = 0.
            utils.color_saveimg(analogy,
                                (sqrtbs, sqrtbs),
                                'imgs/VAE/CVAE_celeba_analogy_epoch' + str((i + 1)) + '.jpg')

            z = f_encode(X_train)[0]
            reconstract = (f_decode(z)[0] + 1) / 2
            reconstract[reconstract > 1] = 1.
            reconstract[reconstract < 0] = 0.
            utils.color_saveimg(reconstract,
                                (sqrtbs, sqrtbs),
                                'imgs/VAE/CVAE_celeba_reconstract_epoch' + str((i + 1)) + '.jpg')
            del reconstract
            del z

if __name__ == '__main__':
    train_cvae_celeba()
