import theano.tensor as T
import theano
import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation, Layer
from ml.deeplearning.optimizers import SGD
from ml.deeplearning.objectives import CrossEntropy
from ml.deeplearning.models import Sequential
from ml import utils
import sys
import timeit


class GaussianSamplingLayer(Layer):
    def __init__(self, n_out):
        self.n_in = None
        self.n_out = n_out
        self.output = None
        self.rng = None

    def set_rng(self, rng):
        self.rng = rng

    def set_input_shape(self, n_in):
        self.n_in = n_in

    def get_output(self, input):
        srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(999999))
        epsilon = srng.normal(size=(input.shape[0], self.n_out), avg=0, std=1., dtype=input.dtype)
        self.output = input[:, :self.n_out] + T.exp(0.5 * input[:, self.n_out:]) * epsilon
        return self.output

    def get_output_train(self, input):
        return self.get_output(input)


class KLD(object):
    def __init__(self, encoder, z_dim, mode=0):
        self.mean = None
        self.log_var = None
        self.encoder = encoder
        self.z_dim = z_dim
        self.mode = mode

    def get_field(self, x):
        output = self.encoder.get_output_train(x)
        self.mean = output[:, :self.z_dim]
        self.log_var = output[:, self.z_dim:]

    def get_output(self):
        output = -T.sum(self.log_var - T.exp(self.log_var) - self.mean ** 2, axis=1)
        if self.mode:
            output = T.sum(output)
        else:
            output = T.mean(output)
        return output

if __name__ == '__main__':
    # load data
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]
    utils.saveimg(X_valid[:100].reshape(100, 28, 28)*255.0, (10, 10), "imgs/VAE/MNIST_Valid.png")
    # set params
    batch_size = 100
    epoch = 500
    rng1 = np.random.RandomState(1)
    z_dim = 10
    # make encoder
    encoder = Sequential(28*28, rng1, iprint=False)
    encoder.add(Dense(500))
    encoder.add(Activation("tanh"))
    encoder.add(Dense(z_dim*2))
    # make decoder
    decoder = Sequential(z_dim, rng1, iprint=False)
    decoder.add(Dense(500))
    decoder.add(Activation("tanh"))
    decoder.add(Dense(28*28))
    decoder.add(Activation("sigmoid"))
    decoder.compile(batch_size=batch_size, nb_epoch=1)
    # concat encoder and decoder
    encoder_decoder = Sequential(28*28, rng1, iprint=False)
    encoder_decoder.add(encoder)
    encoder_decoder.add(GaussianSamplingLayer(z_dim))
    encoder_decoder.add(decoder)
    encoder.compile(batch_size=batch_size, nb_epoch=1)
    # define loss
    encoder_output = encoder.get_top_output_train()
    kld = KLD(encoder=encoder, z_dim=z_dim)
    opt = SGD(lr=0.001, momentum=0.9)
    ce = CrossEntropy(mode=0)
    loss = [ce, kld]

    encoder_decoder.compile(batch_size=batch_size, nb_epoch=1, opt=opt, loss=loss)
    z_plot = np.random.standard_normal((100, z_dim)).astype(np.float32)

    for i in xrange(epoch):
        print "epoch:", i + 1
        s = timeit.default_timer()
        X_train, y_train = utils.shuffle(X_train, y_train)
        encoder_decoder.fit(X_train, X_train)
        e = timeit.default_timer()
        sys.stdout.write('%.2fs' % (e - s))
        sys.stdout.write("\n")
        if (i+1) % 10 == 0:
            generation = 255.0 * decoder.predict(z_plot).reshape(100, 28, 28)
            utils.saveimg(generation, (10, 10), "imgs/VAE/VAE_MNIST_epoch" + str(i+1) + ".png")
            reconstract = 255.0 * encoder_decoder.predict(X_valid[:100]).reshape(100, 28, 28)
            utils.saveimg(reconstract, (10, 10), "imgs/VAE/VAE_MNIST_reconstruct_epoch" + str(i+1) + ".png")
