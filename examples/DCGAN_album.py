import numpy as np
from ml.deeplearning.layers import (Dense, Activation, BatchNormalization,
                                    Flatten, Reshape,  DeconvCUDNN, ConvCUDNN)
from ml.deeplearning.optimizers import Adam
from ml.deeplearning.objectives import CrossEntropy
from ml.deeplearning.models import Sequential
from ml import utils
import sys
import timeit
import os
from StringIO import StringIO
from PIL import Image


def generator(rng, batch_size):
    g = Sequential(100, rng=rng, iprint=False)
    g.add(Dense(512*4*4, init='normal'))
    g.add(BatchNormalization(moving=True, trainable=False))
    g.add(Activation('relu'))
    g.add(Reshape((512, 4, 4)))

    g.add(DeconvCUDNN(256, 4, 4, (256, 8, 8), init='normal',
                      subsample=(2, 2), border_mode=(1, 1)))
    g.add(BatchNormalization(moving=True, trainable=False))
    g.add(Activation('relu'))

    g.add(DeconvCUDNN(128, 4, 4, (128, 16, 16), init='normal',
                      subsample=(2, 2), border_mode=(1, 1)))
    g.add(BatchNormalization(moving=True, trainable=False))
    g.add(Activation('relu'))

    g.add(DeconvCUDNN(64, 4, 4, (64, 32, 32), init='normal',
                      subsample=(2, 2), border_mode=(1, 1)))
    g.add(BatchNormalization(moving=True, trainable=False))
    g.add(Activation('relu'))

    g.add(DeconvCUDNN(3, 4, 4, (3, 64, 64), init='normal',
                      subsample=(2, 2), border_mode=(1, 1)))
    g.add(Activation('tanh'))
    g.compile(batch_size=batch_size, nb_epoch=1)

    return g


def discriminator(rng, batch_size):
    d = Sequential((3, 64, 64), rng=rng, iprint=False)
    d.add(ConvCUDNN(64, 4, 4, init='normal', subsample=(2, 2),
                    border_mode=(1, 1)))
    d.add(Activation('leakyrelu'))

    d.add(ConvCUDNN(128, 4, 4, init='normal', subsample=(2, 2),
                    border_mode=(1, 1)))
    d.add(BatchNormalization(moving=True, trainable=False))
    d.add(Activation('leakyrelu'))

    d.add(ConvCUDNN(256, 4, 4, init='normal', subsample=(2, 2),
                    border_mode=(1, 1)))
    d.add(BatchNormalization(moving=True, trainable=False))
    d.add(Activation('leakyrelu'))

    d.add(ConvCUDNN(512, 8, 8, init='normal', subsample=(2, 2),
                    border_mode=(1, 1)))
    d.add(BatchNormalization(moving=True, trainable=False))
    d.add(Activation('leakyrelu'))

    d.add(Flatten())
    d.add(Dense(1, init='normal'))
    d.add(Activation('sigmoid'))
    d.compile(batch_size=batch_size, nb_epoch=1,
              loss=CrossEntropy(),
              opt=Adam(lr=0.0002, beta1=0.5))

    return d


def samplez(bs, dim):
    z = np.random.uniform(low=-1, high=1, size=bs*dim)
    return z.reshape(bs, dim).astype(np.float32)


def loadimg(filename):
    img = np.asarray(Image.open(StringIO(filename)))
    return img.astype(np.float32).transpose(2, 0, 1)


def train_dcgan_album():
    image_dir = os.path.expanduser('~') + '/dataset/album'
    fs = os.listdir(image_dir)
    dataset = []
    for fn in fs:
        f = open('%s/%s' % (image_dir, fn), 'rb')
        img_bin = f.read()
        dataset.append(img_bin)
        f.close()

    train_data = dataset[:-5000]
    valid_data = dataset[-5000:]
    data_size = len(train_data)
    print data_size

    batch_size = 100
    k = 1
    n_epoch = 100

    rng1 = np.random.RandomState(1)
    d = discriminator(rng1, batch_size)

    rng2 = np.random.RandomState(1234)
    g = generator(rng2, batch_size)

    # concat model for training generator
    concat_g = Sequential(100, rng2, iprint=False)
    concat_g.add(g)
    concat_g.add(d, add_params=False)
    concat_g.compile(batch_size=batch_size, nb_epoch=1,
                     loss=CrossEntropy(),
                     opt=Adam(lr=0.0002, beta1=0.5))

    # make label
    ones = np.ones(batch_size).astype(np.int8)
    zeros = np.zeros(batch_size).astype(np.int8)
    X_train = np.zeros((batch_size, 3, 64, 64), dtype=np.float32)

    # make valid data
    X_valid = np.zeros((5000, 3, 64, 64)).astype(np.float32)
    for l in xrange(5000):
        img = np.asarray(Image.open(StringIO(valid_data[l])).convert('RGB'))
        X_valid[l] = 2*img.astype(np.float32).transpose(2, 0, 1)/255. - 1

    # generate first fake
    for i in range(len(g.layers)):
        if hasattr(g.layers[i], 'moving'):
            g.layers[i].moving = False
    z = samplez(batch_size, 100)
    z_plot = samplez(batch_size, 100)
    fake = g.predict(z)
    for i in range(len(g.layers)):
        if hasattr(g.layers[i], 'moving'):
            g.layers[i].moving = True
    g.pred_function = None

    # train
    for i in xrange(n_epoch):
        start = 0
        print('epoch:{0}'.format(i+1))
        s = timeit.default_timer()
        for j in xrange(data_size/batch_size):
            # load img
            idx = np.random.randint(data_size, size=batch_size)
            for l in xrange(batch_size):
                img = Image.open(StringIO(train_data[idx[l]])).convert('RGB')
                img_arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
                X_train[l] = 2*img_arr/255. - 1
            # train discriminator
            d.onebatch_fit(X_train, ones)
            d.onebatch_fit(fake, zeros)
            # train generator
            if j % k == 0:
                z = samplez(batch_size, 100)
                concat_g.onebatch_fit(z, ones)

            z = samplez(batch_size, 100)
            fake = g.predict(z)
            e1 = timeit.default_timer()
            start += batch_size
            utils.progbar(j+1, data_size/batch_size, e1 - s)

        # evaluation
        z = samplez(5000, 100)
        fake_valid = g.predict(z)
        acc_real = d.accuracy(X_valid, np.ones(5000).astype(np.int8))
        sys.stdout.write(' Real ACC:{0:.3f}'.format(acc_real))
        acc_fake = d.accuracy(fake_valid, np.zeros(5000).astype(np.int8))
        sys.stdout.write(' Gene ACC:{0:.3f}'.format(acc_fake))

        e = timeit.default_timer()
        sys.stdout.write(', {0:.2f}s'.format(e-s))
        sys.stdout.write('\n')

        # generate and save img
        if (i+1) % 1 == 0:
            print('generate fake...')
            generation = 255. * (g.predict(z_plot)+1.) / 2.
            utils.color_saveimg(generation, (10, 10),
                                ('imgs/DCGAN/DCGAN_album_epoch' + str(i+1)
                                 + '.png'))
if __name__ == '__main__':
    train_dcgan_album()
