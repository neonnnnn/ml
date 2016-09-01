import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation, Dropout, Maxout, Conv, Deconv, BatchNormalization, Flatten, Reshape,  DeconvCUDNN
from ml.deeplearning.optimizers import Adam, SGD
from ml.deeplearning.objectives import MulticlassLogLoss, CrossEntropy, L2Regularization
from ml.deeplearning.models import Sequential
from ml import utils

if __name__ == '__main__':
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_train = X_train.reshape((50000, 1, 28, 28)) * 2. - 1
    rng1 = np.random.RandomState(123)

    n_batch = 100
    k = 1
    n_epoch = 100
    # make discriminater
    d = Sequential((1, 28, 28), rng=rng1, iprint=False)
    d.add(Conv(64, 5, 5, subsample=(2, 2), border_mode=(2, 2)))
    d.add(BatchNormalization())
    d.add(Activation("leakyrelu"))
    d.add(Conv(64, 5, 5, subsample=(2, 2), border_mode=(2, 2)))
    d.add(BatchNormalization())
    d.add(Activation("leakyrelu"))
    d.add(Flatten())
    d.add(Dense(1))
    d.add(Activation('sigmoid'))
    d.compile(batch_size=n_batch, nb_epoch=1, loss=CrossEntropy(), opt=Adam(lr=0.0002, beta_1=0.5))

    # make generator
    rng2 = np.random.RandomState(1234)
    g = Sequential(100, rng=rng2, iprint=False)
    g.add(Dense(64*2*7*7))
    g.add(BatchNormalization())
    g.add(Activation("relu"))
    g.add(Reshape((64*2, 7, 7)))
    g.add(DeconvCUDNN(64, 5, 5, (64, 14, 14), subsample=(2, 2), border_mode=(2, 2)))
    g.add(BatchNormalization())
    g.add(Activation("relu"))
    g.add(DeconvCUDNN(1, 5, 5, (1, 28, 28), subsample=(2, 2), border_mode=(2, 2)))
    g.add(Activation("tanh"))
    g.compile(batch_size=n_batch, nb_epoch=1)

    # concat g + d
    concat = Sequential(100, rng2, iprint=False)
    concat.add(g)
    concat.add(d, add_params=False)
    concat.compile(batch_size=n_batch, nb_epoch=1, loss=CrossEntropy(), opt=Adam(lr=0.0002, beta_1=0.5))
    for i in xrange(n_epoch):
        start = 0
        print "epoch:", i+1
        X_train, y_train = utils.shuffle(X_train, y_train)
        for j in xrange(500):
            # generate imitation
            z = np.random.uniform(low=-1, high=1, size=n_batch*100).reshape(n_batch, 100)
            z = z.astype(np.float32)
            imitation = g.predict(z)
            # train discriminator
            d.fit(np.vstack((X_train[start:start + n_batch], imitation)), np.append(np.ones(n_batch), np.zeros(n_batch)).astype(np.int32))
            # train generator
            if j % k == 0:
                z = np.random.uniform(low=-1, high=1, size=n_batch*100).reshape(n_batch, 100).astype(np.float32)
                imitation = g.predict(z)
                print "Real_Acc:", d.accuracy(X_train[start+n_batch:start+2*n_batch], np.ones(n_batch, dtype=np.int32)), \
                    "Imi_Acc:", d.accuracy(imitation, np.zeros(100, dtype=np.int32))
                concat.fit(z, np.ones(n_batch).astype(np.int32))
                imitation = g.predict(z)
                print "Real_Acc:", d.accuracy(X_train[start+n_batch:start+n_batch*2], np.ones(n_batch, dtype=np.int32)), \
                    "Imi_Acc:", d.accuracy(imitation, np.zeros(100, dtype=np.int32))

            start += n_batch

        if (i+1) % 1 == 0:
            print "generate imitation..."
            z = np.random.uniform(low=-1, high=1, size=n_batch*100).reshape(n_batch, 100)
            generation = 255.0 * (g.predict(z.astype(np.float32)).reshape(n_batch, 28, 28) + 1.) / 2.
            utils.visualize(generation, (10, 10), "GAN_MNIST_epoch" + str(i+1) + ".pdf", nomarlization_flag=False)

        if (i+1) % 50 == 0:
            z1 = np.random.uniform(low=-1, high=1, size=100*10).reshape(10, 100)
            z2 = np.random.uniform(low=-1, high=1, size=100*10).reshape(10, 100)
            for i in xrange(10):
                for j in xrange(10):
                    z[i*10+j:i*10+j+1] = z1[i] + j*(z2[i] - z1[i])/9.
            generation = 255.0 * (g.predict(z.astype(np.float32)).reshape(100, 28, 28) + 1.) / 2.
            utils.visualize(generation, figshape=(10, 10), filename="GAN_MNIST_calculation.pdf", nomarlization_flag=False)

    for i in range(len(g.layers)):
        g.save_weights(i, "Weight_"+str(i))



