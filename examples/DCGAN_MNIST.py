import numpy as np
import load_mnist
from ml.deeplearning.layers import (Dense, Activation, Conv,  Flatten,
                                    BatchNormalization, Reshape, DeconvCUDNN)
from ml.deeplearning.optimizers import Adam
from ml.deeplearning.objectives import CrossEntropy, L2Regularization
from ml.deeplearning.models import Sequential
from ml import utils
import sys
import timeit


def generator(rng):
    g = Sequential(100, rng=rng)
    g.add(Dense(64*2*7*7, init='normal'))
    g.add(BatchNormalization(moving=True))
    g.add(Activation('relu'))
    g.add(Reshape((64*2, 7, 7)))
    g.add(DeconvCUDNN(64, 5, 5, (64, 14, 14), init='normal', subsample=(2, 2),
                      border_mode=(2, 2)))
    g.add(BatchNormalization(moving=True))
    g.add(Activation('relu'))
    g.add(DeconvCUDNN(1, 5, 5, (1, 28, 28), init='normal', subsample=(2, 2),
                      border_mode=(2, 2)))
    g.add(Activation('tanh'))

    return g


def discriminator(rng):
    d = Sequential((1, 28, 28), rng=rng)
    d.add(Conv(64, 5, 5, init='normal', subsample=(2, 2), border_mode=(2, 2)))
    d.add(Activation('leakyrelu'))
    d.add(Conv(64, 5, 5, init='normal', subsample=(2, 2), border_mode=(2, 2)))
    d.add(BatchNormalization(moving=True))
    d.add(Activation('leakyrelu'))
    d.add(Flatten())
    d.add(Dense(1, init='normal'))
    d.add(Activation('sigmoid'))
    d.compile(train_loss=[CrossEntropy(), L2Regularization(weight=1e-5)],
              opt=Adam(lr=0.0002, beta1=0.5))

    return d


def train_dcgan_mnist():
    # load data
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]
    X_train = X_train.reshape((50000, 1, 28, 28)) * 2. - 1
    X_valid = X_valid.reshape((10000, 1, 28, 28)) * 2 - 1.
    batch_size = 100
    k = 1
    n_epoch = 200

    filename = 'imgs/DCGAN/DCGAN_MNIST_'
    # make discriminator
    rng1 = np.random.RandomState(1)
    d = discriminator(rng1)

    # make generator
    rng2 = np.random.RandomState(1234)
    g = generator(rng2)

    # concat models for training generator
    concat_g = Sequential(100, rng2)
    concat_g.add(g)
    concat_g.add(d, add_params=False)
    concat_g.compile(train_loss=[CrossEntropy(),
                                 L2Regularization(weight=1e-5)],
                     opt=Adam(lr=0.0002, beta1=0.5))

    # make label
    ones = np.ones(batch_size).astype(np.int8)
    zeros = np.zeros(batch_size).astype(np.int8)

    # generate first fake
    for i, layer in enumerate(g.layers):
        if hasattr(layer, 'moving'):
            g.layers[i].moving = False
    z = np.random.uniform(low=-1, high=1, size=batch_size*100)
    z = z.reshape(batch_size, 100).astype(np.float32)
    fake = g.predict(z)
    for i, layer in enumerate(g.layers):
        if hasattr(layer, 'moving'):
            g.layers[i].moving = True
    g.pred_function = None

    z_plot = np.random.uniform(low=-1, high=1, size=100*100)
    z_plot = z_plot.reshape(batch_size, 100).astype(np.float32)

    # training
    for i in xrange(n_epoch):
        start = 0
        print('epoch:{0}'.format(i+1))
        X_train, y_train = utils.shuffle(X_train, y_train)
        s = timeit.default_timer()
        for j in xrange(50000/batch_size):
            # train discriminator
            d.fit_on_batch(X_train[start:start+batch_size], ones)
            d.fit_on_batch(fake, zeros)

            # train generator
            if j % k == 0:
                z = np.random.uniform(low=-1, high=1, size=batch_size*100)
                z = z.reshape(batch_size, 100).astype(np.float32)
                concat_g.fit_on_batch(z, ones)
            # generate fake
            z = np.random.uniform(low=-1, high=1, size=batch_size*100)
            z = z.reshape(batch_size, 100).astype(np.float32)
            fake = g.predict(z)
            start += batch_size
            e1 = timeit.default_timer()
            utils.progbar(j+1, 50000/batch_size, e1-s)

        # validation
        z = np.random.uniform(low=-1, high=1, size=10000*100)
        z = z.reshape(10000, 100).astype(np.float32)
        fake_valid = g.predict(z)
        acc_real = d.accuracy(X_valid, np.ones(10000).astype(np.int8))
        sys.stdout.write(' Real ACC:{0:.3f}'.format(acc_real))
        acc_fake = d.accuracy(fake_valid, np.zeros(10000).astype(np.int8))
        sys.stdout.write(' Gene ACC:{0:.3f}'.format(acc_fake))

        e = timeit.default_timer()
        sys.stdout.write(', {0:.2f}s'.format(e-s))
        sys.stdout.write('\n')
        print g.layers[1].var_inf.get_value()
        if (i+1) % 10 == 0:
            print('generate fake...')
            generation = 255.0 * (g.predict(z_plot)+1) / 2.
            generation = generation.reshape(100, 28, 28)
            utils.saveimg(generation, (10, 10),
                          filename + 'epoch' + str(i+1) + '.png')

        if (i+1) % 10 == 0:
            z1 = np.random.uniform(low=-1, high=1, size=100)
            z2 = np.random.uniform(low=-1, high=1, size=100)
            z = np.zeros((100, 100))
            for j in xrange(100):
                z[j] = z1 + (-z1 + z2) * j / 99.
            generation = 255.0 * (g.predict(z.astype(np.float32))+1) / 2.
            generation = generation.reshape(100, 28, 28)

            utils.saveimg(generation, (10, 10),
                                filename + 'Analogy_epoch' + str(i+1) + '.png')

if __name__ == '__main__':
    train_dcgan_mnist()
