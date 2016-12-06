import theano.tensor as T
import theano
import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense
from ml.deeplearning.activations import relu, sigmoid
from ml.deeplearning.optimizers import Adam
from ml.deeplearning.models import Model
from ml import utils
from VAE_MNIST import KLD


class CVAE(Model):
    def __init__(self, rng, encoder, decoder, z_dim):
        self.zdim = z_dim
        super(CVAE, self).__init__(rng, encoder=encoder, decoder=decoder)

    def forward(self, x, y, train):
        srng = T.shared_randomstreams.RandomStreams(self.rng.randint(999999))
        encode = self.encoder.forward(x, y, train)
        epsilon = srng.normal(size=(encode.shape[0], self.z_dim), avg=0,
                              std=1., dtype=encode.dtype)
        z = (encode[:, :self.z_dim]
             + epsilon*T.exp(0.5*encode[:, self.z_dim:]))
        decode = self.decoder.forward(z, y, train)
        return encode, decode

    def function(self, opt=None, train=True):
        x = T.matrix('x')
        x = x.reshape((100, 784))
        y = T.matrix('y')
        kld = KLD(self.z_dim, mode=0)
        encode, decode = self.forward(x, y, train)
        if train:
            recons = T.sum(T.nnet.binary_crossentropy(decode, x))
            cost = recons + kld.calc(encode)
            function = theano.function(inputs=[x, y], outputs=[cost],
                                       updates=self.get_updates(cost, opt))
        else:
            function = theano.function(inputs=[x, y], outputs=[decode])

        return function


class Encoder(Model):
    def __init__(self, rng, h1_x, h1_y, h2, h3):
        super(Encoder, self).__init__(rng, h1_x=h1_x, h1_y=h1_y, h2=h2, h3=h3)

    def forward(self, x, y, train):
        h1 = relu(self.h1_x(x)+self.h1_y(y))
        h2 = relu(self.h2(h1))
        output = self.h3(h2)
        return output

    def function(self):
        x = T.matrix('x')
        y = T.matrix('y')
        output = self.forward(x, y, False)
        return theano.function(inputs=[x, y], outputs=[output])


class Dencoder(Model):
    def __init__(self, rng, h1_z, h1_y, h2, h3):
        super(Dencoder, self).__init__(rng, h1_z=h1_z, h1_y=h1_y, h2=h2, h3=h3)

    def forward(self, z, y, train=False):
        h1 = relu(self.h1_z(z)+self.h1_y(y))
        h2 = relu(self.h2(h1))
        output = sigmoid(self.h3(h2))
        return output

    def function(self):
        z = T.matrix('z')
        y = T.matrix('y')
        output = self.forward(z, y, train=False)
        return theano.function(inputs=[z, y], outputs=[output])


def train_cvae_mnist():
    # load data
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]
    y_train = utils.onehot(y_train).astype(np.float32)
    y_valid = utils.onehot(y_valid).astype(np.float32)
    # set params
    batch_size = 100
    epoch = 10000
    rng1 = np.random.RandomState(1)
    z_dim = 50
    # make encoder, decoder
    encoder = Encoder(rng1, h1_x=Dense(500, 784), h1_y=Dense(500, 10),
                      h2=Dense(500, 500), h3=Dense(z_dim*2, 500))
    decoder = Dencoder(rng1, h1_z=Dense(500, z_dim), h1_y=Dense(500, 10),
                       h2=Dense(500, 500), h3=Dense(784, 500))
    # concat encoder and decoder, and define loss
    cvae = CVAE(rng1, encoder=encoder, decoder=decoder, z_dim=z_dim)
    opt = Adam(lr=3e-4)
    f_train = cvae.function(opt, True)
    f_encode = encoder.function()
    f_decode = decoder.function()
    y = utils.onehot((np.arange(100)/10) % 10).astype(np.float32)

    # train
    for i in xrange(epoch):
        print('epoch:{0}'.format(i+1))
        for j in range(y_train.shape[0]/batch_size):
            start = batch_size * j
            end = start + batch_size
            x = rng1.binomial(n=1, p=X_train[start:end]).astype(np.float32)
            f_train(x, y_train[start:end])

        idx = rng1.permutation(y_train.shape[0])
        X_train, y_train = X_train[idx], y_train[idx]
        if (i+1) % 100 == 0:
            z = f_encode(X_valid[:100], y_valid[:100])[0]
            z = np.tile(z[:10, :z_dim], (10, 1)).reshape(100, z_dim)
            reconstract = f_decode(z, y)[0]
            print np.min(reconstract), np.max(reconstract)
            plot = 255 * np.vstack((X_valid[:10], reconstract))
            plot[np.where(plot < 0)] = 0
            plot[np.where(plot > 255)] = 255
            utils.saveimg(plot.reshape(110, 28, 28).astype(np.uint8), (11, 10),
                          ('imgs/CVAE/CVAE_MNIST_reconstruct_epoch'
                           + str((i+1)) + '.png'))
            del plot
            del z

if __name__ == '__main__':
    train_cvae_mnist()
