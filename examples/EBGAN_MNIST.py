import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation
from ml.deeplearning.optimizers import Adam
from ml.deeplearning.models import Sequential
from ml.deeplearning.normalizations import BatchNormalization
from ml.utils import BatchIterator, saveimg
from EBGAN import EBGAN
from ml.deeplearning.initializations import normal
from ml.deeplearning.serializers import save


def train_ebgan_mnist(X_train, z_dim, n_hidden, opt_gen, opt_dis, activation, epoch, batch_size, k,
                      use_PT):

    rng = np.random.RandomState(1)

    generator = Sequential(z_dim, rng)
    generator.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    generator.add(Activation(activation))
    generator.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    generator.add(Activation(activation))
    generator.add(Dense(X_train.shape[1], init=normal(0, 0.001)))
    generator.add(Activation('tanh'))

    encoder = Sequential(X_train.shape[1], rng)
    encoder.add(Dense(n_hidden, init=normal(0, 0.001)))
    encoder.add(Activation(activation))
    encoder.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    encoder.add(Activation(activation))
    encoder.add(Dense(100, init=normal(0, 0.001)))

    decoder = Sequential(100, rng)
    encoder.add(Activation(activation))
    decoder.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    decoder.add(Activation(activation))
    decoder.add(Dense(X_train.shape[1], init=normal(0, 0.001)))
    decoder.add(Activation('tanh'))

    gan = EBGAN(rng, generator=generator, encoder=encoder, decoder=decoder, use_PT=use_PT)
    gan.compile(opt_gen=opt_gen, opt_dis=opt_dis)

    z_plot = np.random.standard_normal((batch_size, z_dim)).astype(np.float32)
    z_batch = BatchIterator([z_plot], batch_size)
    train_batches = BatchIterator([X_train*2. - 1.], batch_size)
    if use_PT:
        filename = 'EBGAN_PT_MNIST'
    else:
        filename = 'EBGAN_MNIST'
    # training
    for i in xrange(epoch/10):
        print('epoch:{0}'.format(i+1))
        gan.fit(train_batches, epoch=10*k, iprint=True)
        generation = 127.5 * (generator.predict(z_batch)+1.)
        saveimg(generation.reshape(100, 28, 28), (10, 10),
                'imgs/EBGAN/'+filename+'_epoch' + str((i + 1)*10) + '.png')

    return gan


def main():
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    batch_size = 100
    z_dim = 50
    n_hidden = 600
    k = 1
    epoch = 200*k
    opt_gen = Adam(lr=2e-4, beta1=0.5)
    opt_dis = Adam(lr=2e-4, beta1=0.5)
    train_ebgan_mnist(X_train, z_dim, n_hidden, opt_gen, opt_dis, 'elu', epoch, batch_size, k,
                      use_PT=False)

if __name__ == '__main__':
    main()