import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation
from ml.deeplearning.optimizers import Adam, RMSprop
from ml.deeplearning.models import Sequential
from ml.deeplearning.normalizations import BatchNormalization
from ml.utils import BatchIterator, saveimg
from WGAN import WGAN
from ml.deeplearning.initializations import normal
from ml.deeplearning.serializers import save


def train_wgan_mnist(X_train, z_dim, n_hidden, opt_gen, opt_dis, activation, epoch, batch_size, k):
    sqrtbs = int(batch_size**0.5)

    rng = np.random.RandomState(1)

    generator = Sequential(z_dim, rng)
    generator.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    generator.add(Activation(activation))
    generator.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    generator.add(Activation(activation))
    generator.add(Dense(X_train.shape[1], init=normal(0, 0.001)))
    generator.add(Activation('tanh'))

    discriminator = Sequential(X_train.shape[1], rng)
    discriminator.add(Dense(n_hidden, init=normal(0, 0.001)))
    discriminator.add(Activation(activation))
    discriminator.add(Dense(n_hidden, init=normal(0, 0.001)))
    discriminator.add(Activation(activation))
    discriminator.add(Dense(100, init=normal(0, 0.001)))

    gan = WGAN(rng, generator=generator, discriminator=discriminator, clipping=(-0.05, 0.05))
    gan.compile(opt_gen=opt_gen, opt_dis=opt_dis)

    z_plot = np.random.standard_normal((batch_size, z_dim)).astype(np.float32)
    z_batch = BatchIterator([z_plot], batch_size)
    train_batches = BatchIterator([X_train*2. - 1.], batch_size, shuffle=True)

    # training
    for i in xrange(epoch/10):
        print('epoch:{0}'.format(i+1))
        gan.fit(train_batches, epoch=10*k, iprint=True, k=k)
        generation = 127.5 * (generator.predict(z_batch)+1.)
        saveimg(generation.reshape(batch_size, 28, 28), (sqrtbs, sqrtbs),
                'imgs/WGAN/WGAN_MNIST_epoch' + str((i + 1)*10) + '.png')
        print(np.max(np.abs(discriminator.layers[0].W.get_value())))

    return gan


def main():
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    batch_size = 64
    z_dim = 100
    n_hidden = 500
    k = 5
    epoch = 500
    opt_gen = RMSprop(lr=0.00005, rho=0.5)
    opt_dis = RMSprop(lr=0.00005, rho=0.5)
    train_wgan_mnist(X_train, z_dim, n_hidden, opt_gen, opt_dis, 'relu', epoch, batch_size, k)

if __name__ == '__main__':
    main()