import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation, Conv, BatchNormalization, Flatten, Reshape, DeconvCUDNN
from ml.deeplearning.optimizers import Adam
from ml.deeplearning.objectives import CrossEntropy, L2Regularization
from ml.deeplearning.models import Sequential
from ml import utils
import sys
import timeit


def generator(rng, batch_size):
    g = Sequential(100, rng=rng, iprint=False)
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
    g.compile(batch_size=batch_size, nb_epoch=1)

    return g


def discriminator(rng, batch_size):
    d = Sequential((1, 28, 28), rng=rng, iprint=False)
    d.add(Conv(64, 5, 5, init='normal', subsample=(2, 2), border_mode=(2, 2)))
    d.add(Activation('leakyrelu'))
    d.add(Conv(64, 5, 5, init='normal', subsample=(2, 2), border_mode=(2, 2)))
    d.add(BatchNormalization(moving=True))
    d.add(Activation('leakyrelu'))
    d.add(Flatten())
    d.add(Dense(1, init='normal'))
    d.add(Activation('sigmoid'))
    d.compile(batch_size=batch_size, nb_epoch=1,
              loss=[CrossEntropy(), L2Regularization(weight=1e-5)],
              opt=Adam(lr=0.0002, beta_1=0.5))

    return d


if __name__ == '__main__':
    # load data
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]
    X_train = X_train.reshape((50000, 1, 28, 28)) * 2. - 1
    X_valid = X_valid.reshape((10000, 1, 28, 28)) * 2 - 1.
    batch_size = 100
    k = 1
    n_epoch = 500

    # make discriminator
    rng1 = np.random.RandomState(1)
    discriminator = discriminator(rng1, batch_size)

    # make generator
    rng2 = np.random.RandomState(1234)
    generator = generator(rng2, batch_size)

    # concat models for training generator
    concat_g = Sequential(100, rng2, iprint=False)
    concat_g.add(generator)
    concat_g.add(discriminator, add_params=False)
    concat_g.compile(batch_size=batch_size, nb_epoch=1, 
                     loss=[CrossEntropy(), L2Regularization(weight=1e-5)], 
                     opt=Adam(lr=0.0002, beta_1=0.5))

    # make label
    ones = np.ones(batch_size).astype(np.int8)
    zeros = np.zeros(batch_size).astype(np.int8)

    # generate first imitation
    for i in range(len(generator.layers)):
        if hasattr(generator.layers[i], 'moving'):
            generator.layers[i].moving = False
    z = np.random.uniform(low=-1, high=1, size=batch_size * 100)
    z = z.reshape(batch_size, 100).astype(np.float32)
    imitation = generator.predict(z)
    for i in range(len(generator.layers)):
        if hasattr(generator.layers[i], 'moving'):
            generator.layers[i].moving = True

    z_plot = np.random.uniform(low=-1, high=1, size=100*100)
    z_plot = z_plot.reshape(batch_size, 100).astype(np.float32)

    # training
    for i in xrange(n_epoch):
        start = 0
        print 'epoch:', i+1
        X_train, y_train = utils.shuffle(X_train, y_train)
        s = timeit.default_timer()
        for j in xrange(50000/batch_size):
            # train discriminator
            s1 = timeit.default_timer()
            discriminator.onebatch_fit(X_train[start:start + batch_size], ones)
            discriminator.onebatch_fit(imitation, zeros)

            # train generator
            if j % k == 0:
                z = np.random.uniform(low=-1, high=1, size=batch_size * 100)
                z = z.reshape(batch_size, 100).astype(np.float32)
                concat_g.onebatch_fit(z, ones)
            # generate imitation
            z = np.random.uniform(low=-1, high=1, size=batch_size * 100)
            z = z.reshape(batch_size, 100).astype(np.float32)
            imitation = generator.predict(z)
            start += batch_size
            e1 = timeit.default_timer()
            utils.progbar(j + 1, 50000 / batch_size, e1 - s1)

        # validation
        z = np.random.uniform(low=-1, high=1, size=10000*100)
        z = z.reshape(10000, 100).astype(np.float32)
        imitation_valid = generator.predict(z)
        real_acc = discriminator.accuracy(X_valid, 
                                          np.ones(10000).astype(np.int8))
        sys.stdout.write(' Real ACC:%.2f' % real_acc)
        imitation_acc = discriminator.accuracy(imitation_valid, 
                                               np.zeros(10000).astype(np.int8))
        sys.stdout.write(' Gene ACC:%.2f' % imitation_acc)

        e = timeit.default_timer()
        sys.stdout.write(', %.2fs' % (e - s))
        sys.stdout.write('\n')

        if (i+1) % 50 == 0:
            print 'generate imitation...'
            generation = 255.0 * generator.predict(z_plot)
            generation = (generation.reshape(100, 28, 28) + 1.) / 2.
            utils.saveimg(generation, (10, 10), 
                          'imgs/DCGAN/DCGAN_MNIST_epoch' + str(i+1) + '.png')

        if (i+1) % 50 == 0:
            z1 = np.random.uniform(low=-1, high=1, size=100)
            z2 = np.random.uniform(low=-1, high=1, size=100)
            z = np.zeros((100, 100))
            for j in xrange(100):
                z[j] = z1 + (-z1 + z2) * j / 99.
            generation = 255.0 * (generator.predict(z.astype(np.float32)))
            generation = (generation.reshape(100, 28, 28) + 1.) / 2.

            utils.color_saveimg(
                generation, (10, 10),
                'imgs/DCGAN/DCGAN_MNIST_Analogy_epoch' + str(i + 1) + '.png')

