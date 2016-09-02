import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation, Dropout, Maxout, Conv, Deconv, BatchNormalization, Flatten, Reshape,  DeconvCUDNN
from ml.deeplearning.optimizers import Adam, SGD
from ml.deeplearning.objectives import MulticlassLogLoss, CrossEntropy, L2Regularization
from ml.deeplearning.models import Sequential
from ml import utils
import sys
import time


def onehot(input, max):
    output = np.zeros(input.shape[0]*max).reshape(input.shape[0], max)
    output[np.arange(input.shape[0]), input] = 1
    return output.astype(np.float32)

if __name__ == '__main__':
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]
    X_train = X_train.reshape((50000, 1, 28, 28)) * 2. - 1
    X_valid = X_valid.reshape((10000, 1, 28, 28)) * 2 - 1.
    rng1 = np.random.RandomState(1)
    n_batch = 100
    k = 1
    n_epoch = 500
    opt = Adam(lr=0.0002, beta_1=0.5)
    # make discriminator
    d = Sequential((1, 28, 28), rng=rng1, iprint=False)
    d.add(Conv(64, 5, 5, init="normal", subsample=(2, 2), border_mode=(2, 2)))
    #d.add(BatchNormalization(moving=True))
    d.add(Activation("leakyrelu"))
    d.add(Conv(64, 5, 5, init="normal", subsample=(2, 2), border_mode=(2, 2)))
    #d.add(BatchNormalization(moving=True))
    d.add(Activation("leakyrelu"))
    d.add(Flatten())
    d.add(Dense(1, init="normal"))
    d.add(Activation('sigmoid'))
    d.compile(batch_size=n_batch, nb_epoch=1, loss=CrossEntropy(), opt=opt)

    # make generator
    rng2 = np.random.RandomState(1234)
    g = Sequential(100, rng=rng2, iprint=False)
    g.add(Dense(64*2*7*7, init="normal"))
    #g.add(BatchNormalization(moving=True))
    g.add(Activation("relu"))
    g.add(Reshape((64*2, 7, 7)))
    g.add(DeconvCUDNN(64, 5, 5, (64, 14, 14), init="normal", subsample=(2, 2), border_mode=(2, 2)))
    g.add(BatchNormalization(moving=True))
    g.add(Activation("relu"))
    g.add(DeconvCUDNN(1, 5, 5, (1, 28, 28), init="normal", subsample=(2, 2), border_mode=(2, 2)))
    g.add(Activation("tanh"))
    g.compile(batch_size=n_batch, nb_epoch=1)

    # concat model for training generator
    concat_g = Sequential(100, rng2, iprint=False)
    concat_g.add(g)
    concat_g.add(d, add_params=False)
    concat_g.compile(batch_size=n_batch, nb_epoch=1, loss=CrossEntropy(), opt=opt)

    ones = np.ones(n_batch).astype(np.int8)
    zeros = np.zeros(n_batch).astype(np.int8)

    for i in xrange(n_epoch):
        start = 0
        print "epoch:", i+1
        X_train, y_train = utils.shuffle(X_train, y_train)
        s = time.time()
        for j in xrange(50000/n_batch):
            # generate imitation
            # train discriminator
            z = np.random.uniform(low=-1, high=1, size=n_batch * 100).reshape(n_batch, 100).astype(np.float32)
            imitation = g.predict(z)
            d.fit(np.vstack((X_train[start:start + n_batch], imitation)), np.append(ones, zeros))
            # train generator
            if j % k == 0:
                z = np.random.uniform(low=-1, high=1, size=n_batch * 100).reshape(n_batch, 100).astype(np.float32)
                concat_g.fit(z, ones)

            start += n_batch
            utils.progbar(j+1, 50000/n_batch)

        z = np.random.uniform(low=-1, high=1, size=10000*100).reshape(10000, 100).astype(np.float32)
        imitation = g.predict(z)
        sys.stdout.write(' Real ACC:%.2f' % d.accuracy(X_valid, np.ones(10000).astype(np.int8)))
        sys.stdout.write(' Gene ACC:%.2f' % d.accuracy(imitation, np.zeros(10000).astype(np.int8)))

        e = time.time()
        sys.stdout.write(', %.2fs' % (e - s))

        sys.stdout.write("\n")

        if (i+1) % 10 == 0:
            print "generate imitation..."
            z = np.random.uniform(low=-1, high=1, size=100*100).reshape(100, 100)
            generation = 255.0 * (g.predict(z.astype(np.float32)).reshape(n_batch, 28, 28) + 1.) / 2.
            utils.saveimg(generation, (10, 10), "imgs/GAN_MNIST_epoch" + str(i+1) + ".png")

    for i in range(len(g.layers)):
        g.save_weights(i, "Weight_"+str(i))
