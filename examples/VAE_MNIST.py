import theano.tensor as T
import theano
import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation
from ml.deeplearning.optimizers import SGD
from ml.deeplearning.objectives import CrossEntropy, Regularization
from ml.deeplearning.models import Sequential, Model, run
from ml import utils
import sys


class KLD(Regularization):
    def __init__(self, z_dim, mode=1):
        super(KLD, self).__init__(weight=1.0)
        self.z_dim = z_dim
        self.mode = mode

    def calc(self, input):
        mean = input[:, :self.z_dim]
        log_var = input[:, self.z_dim:]
        output = -T.sum(1+log_var-T.exp(log_var)-mean**2, axis=1) * 0.5
        if self.mode:
            output = T.mean(output)
        else:
            output = T.sum(output)
        return output


class VAE(Model):
    def __init__(self, rng, encoder, decoder, z_dim):
        self.z_dim = z_dim
        super(VAE, self).__init__(rng, encoder=encoder, decoder=decoder)

    def forward(self, x, train):
        srng = T.shared_randomstreams.RandomStreams(self.rng.randint(999999))
        encode = self.encoder.forward(x, train)
        epsilon = srng.normal(size=(encode.shape[0], self.z_dim), avg=0,
                              std=1., dtype=encode.dtype)
        z = (encode[:, :self.z_dim]
             + epsilon*T.exp(0.5*encode[:, self.z_dim:]))
        decode = self.decoder.forward(z, train)
        return encode, decode

    def function(self, opt=None, train=True):
        x = T.matrix('x')
        x = x.reshape((100, 784))
        kld = KLD(self.z_dim)
        ce = CrossEntropy(mode=1)
        encode, decode = self.forward(x, train)
        if train:
            cost = ce.calc(x, decode) + kld.calc(encode)
            updates = self.updates(cost, opt)
            function = theano.function(inputs=[x], outputs=[cost],
                                       updates=updates)
        else:
            function = theano.function(inputs=[x], outputs=[decode])

        return function


def train_vae_mnist():
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]
    utils.saveimg(X_valid[:100].reshape(100, 28, 28)*255.0, (10, 10),
                  'imgs/VAE/MNIST_Valid.png')

    batch_size = 100
    epoch = 500
    rng = np.random.RandomState(1)
    z_dim = 10

    encoder = Sequential(28*28, rng, iprint=False)
    encoder.add(Dense(500))
    encoder.add(Activation('tanh'))
    encoder.add(Dense(z_dim*2))

    decoder = Sequential(z_dim, rng, iprint=False)
    decoder.add(Dense(500))
    decoder.add(Activation('tanh'))
    decoder.add(Dense(28*28))
    decoder.add(Activation('sigmoid'))
    decoder.compile(batch_size=batch_size, nb_epoch=1)

    vae = VAE(rng, encoder=encoder, decoder=decoder, z_dim=z_dim)
    opt = SGD(lr=0.001, momentum=0.9)
    trainfunction = vae.function(opt, True)
    reconstractfunction = vae.function(opt=None, train=False)
    z_plot = np.random.standard_normal((100, z_dim)).astype(np.float32)
    train_iter = utils.BatchIterator(X_train, 100)

    # train
    for i in xrange(epoch):
        print ('epoch:{0}'.format(i+1))
        run(train_iter, trainfunction, True)
        sys.stdout.write('\n')
        if (i+1) % 500 == 0:
            generation = 255.0 * decoder.predict(z_plot)
            utils.saveimg(generation.reshape(100, 28, 28), (10, 10),
                          'imgs/VAE/VAE_MNIST_epoch' + str(i+1) + '.png')
            reconstract = 255.0 * reconstractfunction(X_valid[:100])
            utils.saveimg(reconstract.reshape(100, 28, 28), (10, 10),
                          ('imgs/VAE/VAE_MNIST_reconstruct_epoch'
                           + str(i+1) + '.png')
                          )

if __name__ == '__main__':
    train_vae_mnist()

