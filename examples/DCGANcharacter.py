import numpy as np
from ml.deeplearning.layers import Dense, Activation, BatchNormalization, Flatten, Reshape,  DeconvCUDNN, ConvCUDNN
from ml.deeplearning.optimizers import Adam
from ml.deeplearning.objectives import CrossEntropy, L2Regularization
from ml.deeplearning.models import Sequential
from ml import utils
import sys
import timeit
import os
from StringIO import StringIO
from PIL import Image


def generator(rng, batch_size):
    g = Sequential(100, rng=rng, iprint=False)
    g.add(Dense(512*4*4, init="normal"))
    g.add(BatchNormalization(moving=True))
    g.add(Activation("relu"))
    g.add(Reshape((512, 4, 4)))
    g.add(DeconvCUDNN(256, 4, 4, (256, 8, 8), init="normal", subsample=(2, 2), border_mode=(1, 1)))
    g.add(BatchNormalization(moving=True))
    g.add(Activation("relu"))
    g.add(DeconvCUDNN(128, 4, 4, (128, 16, 16), init="normal", subsample=(2, 2), border_mode=(1, 1)))
    g.add(BatchNormalization(moving=True))
    g.add(Activation("relu"))
    g.add(DeconvCUDNN(64, 4, 4, (64, 32, 32), init="normal", subsample=(2, 2), border_mode=(1, 1)))
    g.add(BatchNormalization(moving=True))
    g.add(Activation("relu"))
    g.add(DeconvCUDNN(3, 4, 4, (3, 64, 64), init="normal", subsample=(2, 2), border_mode=(1, 1)))
    g.add(Activation("tanh"))
    g.compile(batch_size=batch_size, nb_epoch=1)

    return g


def discriminator(rng, batch_size):
    d = Sequential((3, 64, 64), rng=rng, iprint=False)
    d.add(ConvCUDNN(64, 4, 4, init="normal", subsample=(2, 2), border_mode=(1, 1)))
    d.add(Activation("elu"))
    d.add(ConvCUDNN(128, 4, 4, init="normal", subsample=(2, 2), border_mode=(1, 1)))
    #d.add(BatchNormalization(moving=True))
    d.add(Activation("elu"))
    d.add(ConvCUDNN(256, 4, 4, init="normal", subsample=(2, 2), border_mode=(1, 1)))
    d.add(BatchNormalization(moving=True))
    d.add(Activation("elu"))
    d.add(ConvCUDNN(512, 4, 4, init="normal", subsample=(2, 2), border_mode=(1, 1)))
    d.add(BatchNormalization(moving=True))
    d.add(Activation("elu"))
    d.add(Flatten())
    d.add(Dense(1, init="normal"))
    d.add(Activation('sigmoid'))
    d.compile(batch_size=batch_size, nb_epoch=1, loss=[CrossEntropy(), L2Regularization(1e-5)], opt=Adam(lr=0.0002, beta_1=0.5))

    return d

if __name__ == '__main__':
    # load dataset
    image_dir = os.path.expanduser('~') + "/dataset/3_sv_actors/cropping_rgb/"
    fs = os.listdir(image_dir)
    dataset = []
    for fn in fs:
        f = open('%s/%s' % (image_dir, fn), 'rb')
        img_bin = f.read()
        dataset.append(img_bin)
        f.close()
    valid_size = 1000
    train_data = dataset[:-valid_size]
    valid_data = dataset[valid_size:]
    data_size = len(train_data)
    print data_size

    # set hyper-parameter
    batch_size = 64
    k = 3
    n_epoch = 300

    # make discriminator
    rng1 = np.random.RandomState(1)
    discriminator = discriminator(rng1, batch_size)

    # make generator
    rng2 = np.random.RandomState(1234)
    generator = generator(rng2, batch_size)

    # concat model for training generator
    concat_g = Sequential(100, rng2, iprint=False)
    concat_g.add(generator)
    concat_g.add(discriminator, add_params=False)
    concat_g.compile(batch_size=batch_size, nb_epoch=1, loss=[CrossEntropy(), L2Regularization(1e-5)], opt=Adam(lr=0.0002, beta_1=0.5))

    # make label
    ones = np.ones(batch_size).astype(np.int8)
    zeros = np.zeros(batch_size).astype(np.int8)
    ones2 = np.ones(batch_size * 2).astype(np.int8)
    X_train = np.zeros((batch_size, 3, 64, 64), dtype=np.float32)

    # make valid data
    X_valid = np.zeros((valid_size, 3, 64, 64)).astype(np.float32)
    for l in xrange(valid_size):
        img = np.asarray(Image.open(StringIO(train_data[l]))).astype(np.float32).transpose(2, 0, 1)
        X_valid[l] = img / 255. * 2. - 1.

    # generate first imitation
    for i in range(len(generator.layers)):
        if hasattr(generator.layers[i], "moving"):
            generator.layers[i].moving = False
    z = np.random.uniform(low=-1, high=1, size=batch_size * 100).reshape(batch_size, 100).astype(np.float32)
    imitation = generator.predict(z)
    for i in range(len(generator.layers)):
        if hasattr(generator.layers[i], "moving"):
            generator.layers[i].moving = True

    z_plot = np.random.uniform(low=-1, high=1, size=100 * 100).reshape(100, 100).astype(np.float32)
    # train
    for i in xrange(n_epoch):
        start = 0
        print "epoch:", i+1
        s = timeit.default_timer()
        for j in xrange(data_size/batch_size):
            # load img
            idx = np.random.randint(data_size, size=batch_size)
            for l in xrange(batch_size):
                img = np.asarray(Image.open(StringIO(train_data[idx[l]]))).astype(np.float32).transpose(2, 0, 1)
                X_train[l] = img / 255. * 2. - 1.
            # train discriminator
            discriminator.onebatch_fit(X_train, ones)
            discriminator.onebatch_fit(imitation, zeros)
            # train generator
            if j % k == 0:
                z = np.random.uniform(low=-1, high=1, size=batch_size * 100).reshape(batch_size, 100).astype(np.float32)
                concat_g.onebatch_fit(z, ones)

            z = np.random.uniform(low=-1, high=1, size=batch_size * 100).reshape(batch_size, 100).astype(np.float32)
            concat_g.onebatch_fit(z, ones)

            z = np.random.uniform(low=-1, high=1, size=batch_size * 100).reshape(batch_size, 100).astype(np.float32)
            imitation = generator.predict(z)
            e1 = timeit.default_timer()
            start += batch_size
            utils.progbar(j+1, data_size/batch_size, e1 - s)

        # validation
        z = np.random.uniform(low=-1, high=1, size=valid_size*100).reshape(valid_size, 100).astype(np.float32)
        imitation_valid = generator.predict(z)
        sys.stdout.write(' Real ACC:%.3f' % discriminator.accuracy(X_valid, np.ones(valid_size).astype(np.int8)))
        sys.stdout.write(' Gene ACC:%.3f' % discriminator.accuracy(imitation_valid, np.zeros(valid_size).astype(np.int8)))

        e = timeit.default_timer()
        sys.stdout.write(', %.2fs' % (e - s))
        sys.stdout.write("\n")

        # generate and save img
        if (i+1) % 1 == 0:
            print "generate imitation..."
            generation = 255.0 * (generator.predict(z_plot) + 1.) / 2.
            utils.color_saveimg(generation, (10, 10), "imgs/DCGAN/DCGAN_character_epoch" + str(i+1) + ".png")
        # generate Analogy
        if (i+1) % 10 == 0:
            z1 = np.random.uniform(low=-1, high=1, size=100)
            z2 = np.random.uniform(low=-1, high=1, size=100)
            z = np.zeros((100, 100))
            for j in xrange(100):
                z[j] = z1 + (-z1+z2) * j / 99.
            generation = 255.0 * (generator.predict(z.astype(np.float32)) + 1.) / 2.
            utils.color_saveimg(generation, (10, 10), "imgs/DCGAN/DCGAN_character_Analogy_epoch" + str(i + 1) + ".png")