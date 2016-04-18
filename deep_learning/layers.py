import numpy as np
import theano
import theano.tensor as T
import initializations
import activations
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d


class Dense(object):
    def __init__(self, n_out, W_values=None, b_values=None, init='glorot_uniform'):
        self.have_params = True
        self.n_in = None
        self.n_out = n_out
        self.rng = None
        self.W = None
        self.W_values = W_values
        self.b = None
        self.b_values = b_values
        self.init = initializations.get_init(init)
        self.params = None
        self.fan_in = None
        self.fan_out = None
        self.output = None
        self.output_train = None

    def set_rng(self, rng):
        self.rng = rng

    def set_input_shape(self, n_in):
        self.n_in = n_in

    def set_params(self):
        if self.W_values is None:
            self.fan_in = self.n_in
            self.fan_out = self.n_out
            self.W_values = np.asarray(self.init(self, (self.n_in, self.n_out)), dtype=theano.config.floatX)

        else:   # i.e, add DenseLayer with W
            if not isinstance(self.W, np.ndarray):
                raise Exception("type(W_values) must be numpy.ndarray.")
            if self.W_values.shape != (self.n_in, self.n_out):
                raise Exception("W_values.shape must be (n_in, n_out).")
            if self.W_values.dtype != theano.config.floatX:
                raise Exception("W_values.dtype must be theano.config.floatX.")
        self.W = theano.shared(value=self.W_values, name='W', borrow=True)

        if self.b_values is None:
            self.b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
        else:
            if not isinstance(self.b_values, np.ndarray):
                raise Exception("type(b_values) must be numpy.ndarray.")
            if self.b.shape != (self.n_out,):
                raise Exception("b_values.shape must be (n_out,)")
            if self.b_values.dtype != theano.config.flaotX:
                raise Exception("b_values.dtype must be theano.config.flaotX.")
        self.b = theano.shared(value=self.b_values, name='b', borrow=True)

        self.params = [self.W, self.b]

    def get_output(self, input):
        self.output = T.dot(input, self.W) + self.b
        return self.output
        
    def get_output_train(self, input):
        self.output_train = self.get_output(input)
        return self.output_train


class Activation(object):
    def __init__(self, activation_name, param=None):
        self.have_params = False
        self.n_in = None
        self.n_out = None
        self.afunc = activations.get_activation(activation_name)
        self.param = param
        self.output = None
        self.output_train = None

    def set_input_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def get_output(self, input):
        if self.param is None:
            self.output = self.afunc(input)
        else:
            self.output = self.afunc(input, self.param)
        return self.output

    def get_output_train(self, input):
        self.output_train = self.get_output(input)
        return self.output_train


class BatchNormalization(object):
    # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    # Sergey Ioffe, Christian Szegedy
    # http://arxiv.org/abs/1502.03167
    # This class is incomplete. In above-mentioned paper, mu and sig for inference is estimated by training data, but this code
    # estimated these by inference mini-batch.

    def __init__(self, eps=0.01):
        self.have_params = True
        self.n_in = None
        self.n_out = None
        self.mean = None
        self.var = None
        self.gamma = None
        self.beta = None
        self.params = None
        self.eps = eps
        self.output = None
        self.output_train = None

    def set_input_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def set_params(self):
        gamma_values = np.ones(self.n_in, dtype=theano.config.floatX)
        self.gamma = theano.shared(value=gamma_values, borrow=True)
        beta_values = np.zeros(self.n_out, dtype=theano.config.floatX)
        self.beta = theano.shared(value=beta_values, borrow=True)
        self.params = [self.gamma, self.beta]

    def get_output(self, input):
        self.mean = T.mean(input, axis=0)
        self.var = T.var(input, axis=0)
        self.output = self.gamma * (input - self.mean) / T.sqrt((self.var + self.eps)) + self.beta

        return self.output

    def get_output_train(self, input):
        self.output_train = self.get_output(input)
        return self.output_train


class Dropout(object):
    def __init__(self, p=0.5):
        self.have_params = False
        self.n_in = None
        self.n_out = None
        self.rng = None
        self.p = p
        self.output = None
        self.output_train = None

    def set_rng(self, rng):
        self.rng = rng

    def set_input_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def get_output(self, input):
        self.output = input * (1-self.p)
        return self.output

    def get_output_train(self, input):
        srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(999999))
        self.output_train = input * srng.binomial(size=input.shape, n=1, p=1-self.p, dtype=input.dtype)
        return self.output_train


class Conv(object):
    def __init__(self, nb_filter, nb_height, nb_width, border_mode='valid', init='he_conv_normal', subsample=(1, 1), W_values=None, b_values=None):
        self.have_params = True
        self.n_in = None
        self.n_out = None
        self.rng = None
        self.filter_shape = [nb_filter, None, nb_height, nb_width]
        self.W_values = W_values
        self.b_values = b_values
        self.W = None
        self.b = None
        self.params = None
        self.image_shape = None
        self.fan_in = None
        self.fan_out = None
        self.init = initializations.get_init(init)
        self.border_mode = border_mode
        self.subsample = subsample
        self.output = None
        self.output_train = None

    def set_rng(self, rng):
        self.rng = rng

    def set_input_shape(self, input):
        self.n_in = input
        self.filter_shape[1] = self.n_in[0]
        self.filter_shape = tuple(self.filter_shape)
        if self.border_mode == 'valid':
            fmap_h = self.n_in[1] - self.filter_shape[2] + 1
            fmap_w = self.n_in[2] - self.filter_shape[3] + 1
        elif self.border_mode == 'full':
            fmap_h = self.n_in[1] + self.filter_shape[2] - 1
            fmap_w = self.n_in[2] + self.filter_shape[3] - 1
        elif self.border_mode == 'half':
            fmap_h = self.n_in[1]
            fmap_w = self.n_in[2]
        else:
            raise Exception("border_mode must be valid or full or half.")

        self.n_out = (self.filter_shape[0], fmap_h, fmap_w)

    def set_params(self):
        self.fan_in = np.prod(self.filter_shape[1:])
        self.fan_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])

        if self.W_values is None:
            self.W_values = np.asarray(self.init(self, self.filter_shape), dtype=theano.config.floatX)
        else:  # i.e, add DenseLayer with W
            if not isinstance(self.W, np.ndarray):
                raise Exception("type(W_values) must be numpy.ndarray.")
            if self.W_values.shape != self.filter_shape:
                raise Exception("W_values.shape must be (nb_filter, n_in[0], nb_height, nb_width).")
            if self.W_values.dtype != theano.config.floatX:
                raise Exception("W_values.dtype must be theano.config.floatX.")
        self.W = theano.shared(value=self.W_values, borrow=True)

        if self.b_values is None:
            self.b_values = np.zeros((self.nb_filter_shape[0],), dtype=theano.config.floatX)
        else:
            if not isinstance(self.b_values, np.ndarray):
                raise Exception("Layer-typeError: type(b_values) must be numpy.ndarray.")
            if self.b.shape != (self.nb_filter_shape[0],):
                raise Exception("Layer-shapeError. b_values.shape must be (n_out,)")
            if self.b_values.dtype != theano.config.flaotX:
                raise Exception("Layer-dtypeError. b_values.dtype must be theano.config.flaotX.")
        self.b = theano.shared(value=self.b_values, borrow=True)

        # parameters of the model
        self.params = [self.W, self.b]

    def get_output(self, input):
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=self.filter_shape,
            border_mode=self.border_mode,
            subsample=self.subsample
        )
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.output

    def get_output_train(self, input):
        self.output_train = self.get_output(input)
        return self.output_train


class Pool(object):
    def __init__(self, poolsize=(2, 2)):
        self.have_params = False
        self.n_in = None
        self.n_out = None
        self.poolsize = poolsize
        self.output = None
        self.output_train = None

    def set_input_shape(self, n_in):
        self.n_in = n_in
        self.n_out = (self.n_in[0], self.n_in[1] / self.poolsize[0], self.n_in[2] / self.poolsize[1])

    def get_output(self, input):
        self.output = pool.pool_2d(
                input=input,
                ds=self.poolsize,
                ignore_border=True
        )
        return self.output

    def get_output_train(self, input):
        self.output_train = self.get_output(input)
        return self.output_train


class Flatten(object):
    def __init__(self):
        self.have_params = False
        self.n_in = None
        self.n_out = None
        self.output = None
        self.output_train = None

    def set_input_shape(self, n_in):
        self.n_in = n_in
        self.n_out = self.n_in[0] * self.n_in[1] * self.n_in[2]

    def get_output(self, input):
        self.output = input.flaten(2)
        return self.output

    def get_output_train(self, input):
        self.output_train = self.get_output(input)
        return self.output_train


class GaussianNoise(object):
    def __init__(self, std=0.1):
        self.have_params = False
        self.n_in = None
        self.n_out = None
        self.rng = None
        self.std = std
        self.output = None
        self.output_train = None

    def set_rng(self, rng):
        self.rng = rng

    def set_input_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def get_output(self, input):
        self.output = input
        return self.output

    def get_output_train(self, input):
        srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(999999))
        self.output_train = input + srng.normal(size=input.shape, avg=0, std=self.std,  dtype=input.dtype)
        return self.output_train


class BinominalNoise(object):
    def __init__(self, p=0.3):
        self.have_params = False
        self.n_in = None
        self.n_out = None
        self.rng = None
        self.p = p
        self.output = None
        self.output_train = None

    def set_rng(self, rng):
        self.rng = rng

    def set_input_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def get_output(self, input):
        self.output = input
        return self.output

    def get_output_train(self, input):
        srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(999999))
        self.output_train = input * srng.binomial(size=input.shape, n=1, p=1-self.p, dtype=input.dtype)
        return self.output_train


class Decoder(object):
    def __init__(self, encoder, b_values=None):
        self.have_params = True
        self.n_in = encoder.n_out
        self.n_out = encoder.n_in
        self.W = encoder.W.T
        self.b_values = b_values
        self.b = None
        self.params = []
        self.output = None
        self.output_train = None

    def set_input_shape(self, n_in):
        if n_in != self.n_in:
            raise Exception("Definition of Decoder is wrong.")

    def set_params(self):
        if self.b_values is None:
            self.b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
        else:
            if not isinstance(self.b_values, np.ndarray):
                raise Exception("type(b_values) must be numpy.ndarray.")
            if self.b.shape != (self.n_out,):
                raise Exception("b_values.shape must be (n_out,)")
            if self.b_values.dtype != theano.config.flaotX:
                raise Exception("b_values.dtype must be theano.config.flaotX.")
        self.b = theano.shared(value=self.b_values, name='b', borrow=True)

        self.params = [self.b]

    def get_output(self, input):
        self.output = T.dot(input, self.W) + self.b
        return self.output

    def get_output_train(self, input):
        self.output_train = self.get_output(input)
        return self.output_train


