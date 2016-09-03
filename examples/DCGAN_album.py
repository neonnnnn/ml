import numpy as np
from ml.deeplearning.layers import Dense, Activation, Conv, BatchNormalization, Flatten, Reshape,  DeconvCUDNN
from ml.deeplearning.optimizers import Adam
from ml.deeplearning.objectives import CrossEntropy, L2Regularization
from ml.deeplearning.models import Sequential
from ml import utils
import sys
import time
import os
from StringIO import StringIO
from PIL import Image


def onehot(input, max):
    output = np.zeros(input.shape[0]*max).reshape(input.shape[0], max)
    output[np.arange(input.shape[0]), input] = 1
    return output.astype(np.float32)


def generator(rng, batch_size):
    g = Sequential(100, rng=rng, iprint=False)
    g.add(Dense(256*8*8, init="normal"))
    g.add(BatchNormalization(moving=True))
    g.add(Activation("relu"))
    g.add(Reshape((256, 8, 8)))
    g.add(DeconvCUDNN(128, 4, 4, (128, 16, 16), init="normal", subsample=(2, 2), border_mode=(1, 1)))
    g.add(BatchNormalization(moving=True))
    g.add(Activation("relu"))
    g.add(DeconvCUDNN(64, 4, 4, (64, 32, 32), init="normal", subsample=(2, 2), border_mode=(1, 1)))
    g.add(BatchNormalization(moving=True))
    g.add(Activation("relu"))
    g.add(DeconvCUDNN(3, 4, 4, (3, 64, 64), init="normal", subsample=(2, 2), border_mode=(1, 1)))
    g.add(Activation("tanh"))
    g.compile(batch_size=batch_size, nb_epoch=1, loss=[CrossEntropy(), L2Regularization(1e-5)], opt=Adam(lr=0.0002, beta_1=0.5))

    return g


def discriminator(rng, batch_size):
    d = Sequential((3, 64, 64), rng=rng, iprint=False)
    d.add(Conv(64, 4, 4, init="normal", subsample=(2, 2), border_mode=(1, 1)))
    d.add(Activation("elu"))
    d.add(Conv(128, 4, 4, init="normal", subsample=(2, 2), border_mode=(1, 1)))
    d.add(BatchNormalization(moving=True))
    d.add(Activation("elu"))
    d.add(Conv(256, 4, 4, init="normal", subsample=(2, 2), border_mode=(1, 1)))
    d.add(BatchNormalization(moving=True))
    d.add(Activation("elu"))
    d.add(Flatten())
    d.add(Dense(1, init="normal"))
    d.add(Activation('sigmoid'))
    d.compile(batch_size=batch_size, nb_epoch=1, loss=[CrossEntropy(), L2Regularization(1e-5)], opt=Adam(lr=0.0002, beta_1=0.5))

    return d

if __name__ == '__main__':
    # load dataset
    image_dir = os.path.expanduser('~') + "/dataset/album"
    fs = os.listdir(image_dir)
    print len(fs)
    dataset = []
    for fn in fs:
        f = open('%s/%s' % (image_dir, fn), 'rb')
        img_bin = f.read()
        dataset.append(img_bin)
        f.close()

    data_size = len(dataset)
    print data_size

    batch_size = 100
    k = 1
    n_epoch = 300
    # make discriminator
    rng1 = np.random.RandomState(1)
    d = discriminator(rng1, batch_size)

    # make generator
    rng2 = np.random.RandomState(1234)
    g = generator(rng2, batch_size)

    # concat model for training generator
    concat_g = Sequential(100, rng2, iprint=False)
    concat_g.add(g)
    concat_g.add(d, add_params=False)
    concat_g.compile(batch_size=batch_size, nb_epoch=1, loss=[CrossEntropy(), L2Regularization(1e-5)], opt=Adam(lr=0.0002, beta_1=0.5))

    # make label
    ones = np.ones(batch_size).astype(np.int8)
    zeros = np.zeros(batch_size).astype(np.int8)
    X_train = np.zeros((batch_size, 3, 64, 64), dtype=np.float32)

    # generate first imitation
    for i in range(len(g.layers)):
        if hasattr(g.layers[i], "moving"):
            g.layers[i].moving = False

    z = np.random.uniform(low=-1, high=1, size=batch_size * 100).reshape(batch_size, 100).astype(np.float32)
    imitation = g.predict(z)

    for i in range(len(g.layers)):
        if hasattr(g.layers[i], "moving"):
            g.layers[i].moving = True

    for i in xrange(n_epoch):
        start = 0
        print "epoch:", i+1
        s = time.time()
        for j in xrange(data_size/batch_size):
            # load img
            idx = np.random.randint(len(dataset), size=batch_size)
            for l in xrange(batch_size):
                img = np.asarray(Image.open(StringIO(dataset[idx[l]])).convert('RGB')).astype(np.float32).transpose(2, 0, 1)
                X_train[l] = img / 255. * 2. - 1.

            # train discriminator
            # d.fit(np.vstack((X_train, imitation)), np.append(ones, zeros))
            d.fit(X_train, ones)
            d.fit(imitation, zeros)
            # train generator
            if j % k == 0:
                z = np.random.uniform(low=-1, high=1, size=batch_size * 100).reshape(batch_size, 100).astype(np.float32)
                concat_g.fit(z, ones)

            start += batch_size

            z = np.random.uniform(low=-1, high=1, size=batch_size * 100).reshape(batch_size, 100).astype(np.float32)
            imitation = g.predict(z)
            utils.progbar(j+1, data_size/batch_size)

        e = time.time()
        sys.stdout.write(', %.2fs' % (e - s))

        sys.stdout.write("\n")

        if (i+1) % 1 == 0:
            print "generate imitation..."
            z = np.random.uniform(low=-1, high=1, size=400*100).reshape(400, 100)
            generation = 255.0 * (g.predict(z.astype(np.float32)) + 1.) / 2.
            utils.color_saveimg(generation, (20, 20), "imgs/DCGAN_album_epoch" + str(i+1) + ".png")

        if (j+1) % 50 == 0:
            z1 = np.random.uniform(low=-1, high=1, size=100)
            z2 = np.random.uniform(low=-1, high=1, size=100)
            z = np.random.zeros((100, 100))
            for j in xrange(100):
                z[j] = z1 + (-z1+z2) * j / 99.
            generation = 255.0 * (g.predict(z.astype(np.float32)) + 1.) / 2.
            utils.color_saveimg(generation, (20, 20), "imgs/DCGAN_album_Analogy_epoch" + str(i + 1) + ".png")

    for i in range(len(g.layers)):
        g.save_weights(i, "album_generator_Weight_"+str(i))



