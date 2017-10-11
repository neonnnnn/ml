from VAE import VAE
import numpy as np
import os
from ml.deeplearning.layers import (Dense, BatchNormalization, Flatten,
                                    Reshape, DeconvCUDNN, ConvCUDNN,
                                    Activation)
from ml.deeplearning.optimizers import Adam
from ml.deeplearning.models import Model, Sequential
from ml.deeplearning.distributions import Gaussian
from ml.deeplearning import run_on_batch, variable
from ml import utils
from StringIO import StringIO
from PIL import Image


def x_encoder(rng):
    clf = Sequential((3, 64, 64), rng=rng)

    clf.add(ConvCUDNN(32, 4, 4, subsample=(2, 2),
                      border_mode=(1, 1)))
    #clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(ConvCUDNN(64, 4, 4, subsample=(2, 2),
                      border_mode=(1, 1)))
    #clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(ConvCUDNN(128, 4, 4, subsample=(2, 2),
                      border_mode=(1, 1)))
    #clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(ConvCUDNN(256, 4, 4, subsample=(2, 2),
                      border_mode=(1, 1)))
    #clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Flatten())
    clf.add(Dense(512))
    #clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    return clf


def z_decoder(rng, z_dim):
    g = Sequential(z_dim, rng=rng)
    g.add(Dense(256 * 2 * 2))
    g.add(Activation('relu'))
    g.add(Reshape((256, 2, 2)))

    g.add(DeconvCUDNN(256, 4, 4, (256, 4, 4),
                      subsample=(2, 2), border_mode=(1, 1)))
    #g.add(BatchNormalization())
    g.add(Activation('relu'))

    g.add(DeconvCUDNN(256, 4, 4, (256, 8, 8),
                      subsample=(2, 2), border_mode=(1, 1)))
    #g.add(BatchNormalization())
    g.add(Activation('relu'))

    g.add(DeconvCUDNN(128, 4, 4, (128, 16, 16),
                      subsample=(2, 2), border_mode=(1, 1)))
    #g.add(BatchNormalization())
    g.add(Activation('relu'))

    g.add(DeconvCUDNN(64, 4, 4, (64, 32, 32),
                      subsample=(2, 2), border_mode=(1, 1)))
    #g.add(BatchNormalization())
    g.add(Activation('relu'))

    return g


def loadimg(filename):
    img = np.asarray(Image.open(StringIO(filename)))
    return img.astype(np.float32).transpose(2, 0, 1)


def train_vae_character():
    # set params
    batch_size = 100
    sqrtbs = int(batch_size ** 0.5)
    epoch = 200
    rng = np.random.RandomState(1)

    z_dim = 128
    # load data
    image_dir = os.path.expanduser('~') + '/dataset/3_sv_actors/cropping_rgb/'
    fs = os.listdir(image_dir)
    dataset = []
    for fn in fs:
        f = open(image_dir+'/'+fn, 'rb')
        img_bin = f.read()
        dataset.append(img_bin)
        f.close()
    n_data = len(dataset)
    print(n_data)
    train_data = dataset[:-batch_size]
    valid_data = dataset[-batch_size:]
    X_train = np.zeros((batch_size, 3, 64, 64), dtype=np.float32)
    X_valid = np.zeros((batch_size, 3, 64, 64), dtype=np.float32)

    for l in xrange(batch_size):
        img = loadimg(valid_data[l])
        X_valid[l] = 2*img/255. - 1

    z_plot = np.random.standard_normal((batch_size, z_dim)).astype(np.float32)
    z_plot[-1] = -z_plot[0]
    for i in range(batch_size):
        z_plot[i] = ((batch_size-i)*z_plot[0] + i*z_plot[-1])/batch_size
    # train
    valid_batches = utils.BatchIterator(X_valid, batch_size)
    z_batches = utils.BatchIterator(z_plot, batch_size)

    # make encoder, decoder, discriminator, and cvaegan
    print('making encoder ...')
    encoder_x = x_encoder(rng)
    encoder_x.predict(valid_batches)
    encoder_mean = Dense(z_dim)
    encoder_var = Dense(z_dim)
    encoder = Gaussian(encoder_mean, encoder_var, encoder_x,)

    print('making decoder ...')
    decoder_mean = DeconvCUDNN(3, 4, 4, n_out=(3, 64, 64), n_in=(64, 32, 32),
                               subsample=(2, 2), border_mode=(1, 1))
    decoder_logvar = DeconvCUDNN(3, 4, 4, n_out=(3, 64, 64), n_in=(64, 32, 32),
                                 subsample=(2, 2), border_mode=(1, 1))
    decoder_z = z_decoder(rng, z_dim)
    decoder = Gaussian(decoder_mean, decoder_logvar, decoder_z)

    vae = VAE(rng, encoder=encoder, decoder=decoder)
    opt = Adam(lr=1e-4)
    vae.compile(opt, train_loss=None)
    print('making function ...')

    train_function = vae.function(variable(X_train), mode='train')
    utils.color_saveimg(X_valid, (sqrtbs, sqrtbs),
                        ('imgs/VAE/VAE_character_reconstract_epoch'
                         + str((i + 1)) + '.jpg'))
    for i in xrange(epoch):
        print('epoch:{0}'.format(i + 1))
        index = np.random.permutation(len(train_data))
        for j, idx in enumerate(index):
            img = loadimg(train_data[idx])
            X_train[(j+1) % batch_size] = 2.*img/255. - 1.0
            if (j+1) % batch_size == 0:
                run_on_batch([X_train], train_function)

        if (i + 1) % 1 == 0:
            analogy = (decoder.predict(z_batches) + 1.) / 2.
            analogy[analogy > 1] = 1.
            analogy[analogy < 0] = 0.
            utils.color_saveimg(analogy, (sqrtbs, sqrtbs),
                                ('imgs/VAE/VAE_character_analogy_epoch'
                                 + str((i + 1)) + '.jpg'))

            reconstract = (vae.predict(valid_batches) + 1.) / 2.
            reconstract[reconstract > 1] = 1.
            reconstract[reconstract < 0] = 0.
            utils.color_saveimg(reconstract, (sqrtbs, sqrtbs),
                                ('imgs/VAE/VAE_character_reconstract_epoch'
                                 + str((i + 1)) + '.jpg'))

if __name__ == '__main__':
    train_vae_character()
