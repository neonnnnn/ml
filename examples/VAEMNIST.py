import theano.tensor as T
import theano
import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation, Layer
from ml.deeplearning.optimizers import SGD
from ml.deeplearning.objectives import CrossEntropy, Regularization
from ml.deeplearning.models import Sequential, Model, run
from ml import utils
import sys


class GaussianSamplingLayer(Layer):
    def __init__(self, n_out, n_in):
        self.n_in = n_in
        self.n_out = n_out
        self.rng = None

    def set_rng(self, rng):
        self.rng = rng

    def set_input_shape(self, n_in):
        self.n_in = n_in

    def forward(self, input, train=True):
        srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(999999))
        epsilon = srng.normal(size=(input.shape[0], self.n_out), avg=0, std=1., dtype=input.dtype)
        output = input[:, :self.n_out] + T.exp(0.5 * input[:, self.n_out:]) * epsilon
        return output


class KLD(Regularization):
    def __init__(self, z_dim, mode=1):
        super(KLD, self).__init__(weight=1.0)
        self.z_dim = z_dim
        self.mode = mode

    def calc(self, input):
        mean = input[:, :self.z_dim]
        log_var = input[:, self.z_dim:]
        output = -T.sum(log_var - T.exp(log_var) - mean ** 2, axis=1)
        if self.mode:
            output = T.mean(output)
        else:
            output = T.sum(output)
        return output


class VAE(Model):
    def __init__(self, rng, encoder, decoder, samplinglayer):
        super(VAE, self).__init__(rng, encoder=encoder, decoder=decoder, samplinglayer=samplinglayer)

    def forward(self, x, train):
        encoder_output = self.encoder.forward(x, train)
        sampling_output = self.samplinglayer.forward(encoder_output, train)
        decoder_output = self.decoder.forward(sampling_output, train)
        return encoder_output, decoder_output

    def function(self, opt=None, train=True):
        x = T.matrix('x')
        x = x.reshape((100, 784))
        kld = KLD(10)
        ce = CrossEntropy(mode=1)

        encoder_output, decoder_output = self.forward(x, train)
        if train:
            cost = ce.calc(x, decoder_output) + kld.calc(encoder_output)
            updates = self.updates(cost, opt)
            function = theano.function(inputs=[x], outputs=cost, updates=updates)
        else:
            function = theano.function(inputs=[x], outputs=decoder_output)

        return function

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
    # define loss
    opt = SGD(lr=0.001, momentum=0.9)
    vae = VAE(rng1, encoder=encoder, decoder=decoder, samplinglayer=GaussianSamplingLayer(n_in=2*z_dim, n_out=z_dim))

    trainfunction = vae.function(opt, True)
    reconstractfunction = vae.function(opt=None, train=False)
    z_plot = np.random.standard_normal((100, z_dim)).astype(np.float32)
    train_iter = utils.BatchIterator(X_train, 100)
    for i in xrange(epoch):
        print "epoch:", i + 1
        X_train, y_train = utils.shuffle(X_train, y_train)
        run(train_iter, trainfunction, True)
        bs = 0
        sys.stdout.write("\n")
        if (i+1) % 500 == 0:
            generation = 255.0 * decoder.predict(z_plot).reshape(100, 28, 28)
            utils.saveimg(generation, (10, 10), "imgs/VAE/VAE_MNIST_epoch" + str(i+1) + ".png")
            reconstract = 255.0 * reconstractfunction(X_valid[:100]).reshape(100, 28, 28)
            utils.saveimg(reconstract, (10, 10), "imgs/VAE/VAE_MNIST_reconstruct_epoch" + str(i+1) + ".png")
