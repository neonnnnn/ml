import numpy as np
import theano
import theano.tensor as T
import initializations
import activations
import inspect
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from theano.sandbox.cuda.dnn import dnn_conv, GpuDnnConvDesc, GpuDnnConvGradI
from theano.sandbox.cuda.basic_ops import gpu_alloc_empty, gpu_contiguous


class Dense(object):
    def __init__(self, n_out, W_values=None, b_values=None, init='glorot_uniform'):
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
                raise TypeError("type(W_values) must be numpy.ndarray.")
            if self.W_values.shape != (self.n_in, self.n_out):
                raise ValueError("W_values.shape must be (n_in, n_out).")
            if self.W_values.dtype != theano.config.floatX:
                raise ValueError("W_values.dtype must be theano.config.floatX.")
        self.W = theano.shared(value=self.W_values, name='W', borrow=True)

        if self.b_values is None:
            self.b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
        else:
            if not isinstance(self.b_values, np.ndarray):
                raise TypeError("type(b_values) must be numpy.ndarray.")
            if self.b.shape != (self.n_out,):
                raise TypeError("b_values.shape must be (n_out,)")
            if self.b_values.dtype != theano.config.flaotX:
                raise ValueError("b_values.dtype must be theano.config.flaotX.")
        self.b = theano.shared(value=self.b_values, name='b', borrow=True)

        self.params = [self.W, self.b]

    def get_output(self, input):
        self.output = T.dot(input, self.W) + self.b
        return self.output
        
    def get_output_train(self, input):
        self.output_train = T.dot(input, self.W) + self.b
        return self.output_train


class Activation(object):
    def __init__(self, activation_name, *args):
        self.n_in = None
        self.n_out = None
        self.afunc = activations.get_activation(activation_name)
        self.act_param = args
        self.output = None
        self.output_train = None

    def set_input_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def get_output(self, input):
        if len(inspect.getargspec(self.afunc)[0]) > 1 and len(self.act_param) != 0:
            self.output = self.afunc(input, self.act_param)
        else:
            self.output = self.afunc(input)
        return self.output

    def get_output_train(self, input):
        self.output_train = self.get_output(input)
        return self.output_train


class BatchNormalization(object):
    def __init__(self, eps=1e-5, trainable=True, momentum=0.9, moving=True):
        self.n_in = None
        self.n_out = None
        self.mean_inf = None
        self.var_inf = None
        self.gamma = 1.0
        self.beta = 0.0
        self.params = None
        self.eps = eps
        self.output = None
        self.output_train = None
        self.momentum = momentum
        self.trainable = trainable
        self.moving = moving
        self.updates = None

    def set_input_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def set_params(self):
        if type(self.n_in) == int:
            shape = self.n_in
        else:
            shape = self.n_in[0]
        if self.moving:
            self.mean_inf = theano.shared(value=np.zeros(shape, dtype=theano.config.floatX), borrow=True)
            self.var_inf = theano.shared(value=np.zeros(shape, dtype=theano.config.floatX), borrow=True)

        self.gamma = theano.shared(value=np.ones(shape, dtype=theano.config.floatX), borrow=True)
        self.beta = theano.shared(value=np.zeros(shape, dtype=theano.config.floatX), borrow=True)

        if self.trainable:
            self.params = [self.gamma, self.beta]
        else:
            self.params = []

        self.momentum = theano.shared(np.asarray(self.momentum, dtype=theano.config.floatX), borrow=True)

    def get_output(self, input):
        if self.moving:
            if input.ndim == 2:
                self.output = self.gamma * (input - self.mean_inf)
                self.output /= T.sqrt(self.var_inf + self.eps) + self.beta
            elif input.ndim == 4:
                self.output = self.gamma.dimshuffle('x', 0, 'x', 'x') * (input - self.mean_inf.dimshuffle('x', 0, 'x', 'x'))
                self.output /= T.sqrt(self.var_inf + self.eps).dimshuffle('x', 0, 'x', 'x') + self.beta.dimshuffle('x', 0, 'x', 'x')
        else:
            self.output = self.get_output_train(input)

        return self.output

    def get_output_train(self, input):
        if input.ndim == 2:
            mean = T.mean(input, axis=0)
            var = T.var(input, axis=0)
            self.output_train = self.gamma * (input - mean) / T.sqrt(var + self.eps) + self.beta
            self.output_train = T.nnet.batch_normalization(input, self.gamma, self.beta, mean, T.sqrt(var+self.eps))
        elif input.ndim == 4:
            mean = T.mean(input, axis=(0, 2, 3))
            var = T.var(input, axis=(0, 2, 3))
            self.output_train = self.gamma.dimshuffle('x', 0, 'x', 'x') * (input - mean.reshape((1, input.shape[1], 1, 1)))
            self.output_train /= T.sqrt(var + self.eps).reshape((1, input.shape[1], 1, 1)) + self.beta.dimshuffle('x', 0, 'x', 'x')

        if self.moving:
            bs = input.shape[0].astype(theano.config.floatX)
            self.updates = [(self.mean_inf, self.momentum * self.mean_inf + (1 - self.momentum) * mean),
                            (self.var_inf, self.momentum * self.var_inf + (1 - self.momentum) * var * bs / (bs - 1.))]
        else:
            self.updates = []

        return self.output_train


class Dropout(object):
    def __init__(self, p=0.5):
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

    def set_input_shape(self, n_in):
        self.n_in = n_in
        self.filter_shape[1] = self.n_in[0]
        self.filter_shape = tuple(self.filter_shape)
        if self.border_mode == 'valid':
            pad = [0, 0]
        elif self.border_mode == 'full':
            pad = [self.filter_shape[1] - 1, self.filter_shape[2] - 1]
        elif self.border_mode == 'half':
            pad = [self.filter_shape[1] // 2, self.filter_shape[2] // 2]
        elif isinstance(self.border_mode, (list, tuple)):
            pad = self.border_mode
        elif isinstance(self.border_mode, int):
            if self.border_mode < 0:
                raise ValueError("border_mode must be >= 0.")
            pad = [self.border_mode, self.border_mode]
        else:
            raise ValueError('invalid border_mode {}, '
                             'which must be either "valid", "full", "half", an integer or a pair of integers'.format(self.border_mode))
        fmap_h = (self.n_in[1] - self.filter_shape[2] + 2 * pad[0]) // self.subsample[0] + 1
        fmap_w = (self.n_in[2] - self.filter_shape[3] + 2 * pad[1]) // self.subsample[1] + 1

        self.n_out = (self.filter_shape[0], fmap_h, fmap_w)

    def set_params(self):
        self.fan_in = np.prod(self.filter_shape[1:])
        self.fan_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])

        if self.W_values is None:
            self.W_values = np.asarray(self.init(self, self.filter_shape), dtype=theano.config.floatX)
        else:  # i.e, add DenseLayer with W
            if not isinstance(self.W, np.ndarray):
                raise TypeError("type(W_values) must be numpy.ndarray.")
            if self.W_values.shape != self.filter_shape:
                raise ValueError("W_values.shape must be (nb_filter, n_in[0], nb_height, nb_width).")
            if self.W_values.dtype != theano.config.floatX:
                raise ValueError("W_values.dtype must be theano.config.floatX.")
        self.W = theano.shared(value=self.W_values, borrow=True)

        if self.b_values is None:
            self.b_values = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        else:
            if not isinstance(self.b_values, np.ndarray):
                raise Exception("Layer-typeError: type(b_values) must be numpy.ndarray.")
            if self.b.shape != (self.nb_filter_shape[0],):
                raise Exception("Layer-shapeError. b_values.shape must be (n_out,)")
            if self.b_values.dtype != theano.config.flaotX:
                raise Exception("Layer-dtypeError. b_values.dtype must be theano.config.flaotX.")
        self.b = theano.shared(value=self.b_values, borrow=True)

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


class Deconv(Conv):
    def __init__(self, nb_filter, nb_height, nb_width, output_shape, border_mode='adapt', init='he_conv_normal', subsample=(1, 1),
                 W_values=None, b_values=None):
        self.n_in = None
        self.n_out = output_shape
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

    def set_params(self):
        self.fan_in = np.prod(self.filter_shape[1:])
        self.fan_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])

        if self.W_values is None:
            self.W_values = np.asarray(self.init(self, self.filter_shape), dtype=theano.config.floatX)
            self.W_values = self.W_values.reshape((self.filter_shape[1], self.filter_shape[0], self.filter_shape[2], self.filter_shape[3]))
        else:  # i.e, add DenseLayer with W
            if not isinstance(self.W, np.ndarray):
                raise TypeError("type(W_values) must be numpy.ndarray.")
            if self.W_values.shape != (self.filter_shape[1], self.filter_shape[0], self.filter_shape[2], self.filter_shape[3]):
                raise ValueError("W_values.shape must be (nb_filter, n_in[0], nb_height, nb_width).")
            if self.W_values.dtype != theano.config.floatX:
                raise ValueError("W_values.dtype must be theano.config.floatX.")
        self.W = theano.shared(value=self.W_values, borrow=True)

        if self.b_values is None:
            self.b_values = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        else:
            if not isinstance(self.b_values, np.ndarray):
                raise TypeError("Layer-typeError: type(b_values) must be numpy.ndarray.")
            if self.b.shape != (self.nb_filter_shape[0],):
                raise ValueError("Layer-shapeError. b_values.shape must be (n_out,)")
            if self.b_values.dtype != theano.config.flaotX:
                raise ValueError("Layer-dtypeError. b_values.dtype must be theano.config.flaotX.")
        self.b = theano.shared(value=self.b_values, borrow=True)

        self.params = [self.W, self.b]

    def set_rng(self, rng):
        self.rng = rng

    def set_input_shape(self, n_in):
        self.n_in = n_in
        self.filter_shape[1] = self.n_in[0]
        self.filter_shape = tuple(self.filter_shape)

        if self.n_out[0] != self.filter_shape[0]:
            raise ValueError("output_shape[0] must be equal nb_filter.")

        if self.border_mode == 'valid':
            pad = [0, 0]
        elif self.border_mode == 'full':
            pad = [self.filter_shape[1] - 1, self.filter_shape[2] - 1]
        elif self.border_mode == 'half':
            pad = [self.filter_shape[1] // 2, self.filter_shape[2] // 2]
        elif isinstance(self.border_mode, (list, tuple)):
            pad = self.border_mode
        elif isinstance(self.border_mode, int):
            if self.border_mode < 0:
                raise ValueError("border_mode must be >= 0.")
            pad = [self.border_mode, self.border_mode]
        elif self.border_mode == "adapt":
            pad = [0, 0]
            inf = np.array(self.subsample) * (np.array(n_in[1:]) - 1) + np.array(self.filter_shape[2:])
            sup = inf + np.array(self.subsample) - 1
            if self.n_out[1] > sup[0]:
                raise ValueError('border_mode is "adapt" but output_shape[1] is impossible. filter_size or subsample must be larger.')
            elif self.n_out[2] > sup[1]:
                raise ValueError('border_mode is "adapt" but output_shape[2] is impossible. filter_size or subsample must be larger.')
            elif self.n_out[1] < inf[0] & self.n_out[2] < inf[1]:
                pad[0] = (inf[0] - self.n_out[1]) // 2
                pad[1] = (inf[1] - self.n_out[2]) // 2
                self.border_mode = pad
        else:
            raise ValueError('invalid border_mode {}, '
                             'which must be either "valid", "full", "half", an integer or a pair of integers'.format(self.border_mode))

        inf = np.array(self.subsample) * (np.array(n_in[1:]) - 1) + np.array(self.filter_shape[2:]) - 2 * np.array(pad)
        sup = inf + np.array(self.subsample) - 1
        if not ((inf[0] <= self.n_out[1] <= sup[0]) and (inf[1] <= self.n_out[2] <= sup[1])):
            raise ValueError("impossible output_shape. output = subsample * (input - 1) + filter - 2 * pad + a, a \in {0, \ldots , subsample-1}")

    def get_output(self, input):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=(None, self.n_out[0], self.n_out[1], self.n_out[2]),
            kshp=self.filter_shape,
            subsample=self.subsample,
            border_mode=self.border_mode,
            filter_flip=True
        )
        deconv_out = op(self.W, input, self.n_out[1:])
        self.output = deconv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.output

    def get_output_train(self, input):
        self.output_train = self.get_output(input)
        return self.output_train


class ConvCUDNN(Conv):
    def get_output(self, input):
        conv_out = dnn_conv(
            img=input,
            kerns=self.W,
            border_mode=self.border_mode,
            subsample=self.subsample
        )
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.output


class DeconvCUDNN(Deconv):
    def get_output(self, input):
        img = gpu_contiguous(input)
        kerns = gpu_contiguous(self.W)
        desc = GpuDnnConvDesc(border_mode=self.border_mode, subsample=self.subsample,
                              conv_mode="conv")(
            gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2] * self.subsample[0], img.shape[3] * self.subsample[1]).shape, kerns.shape)
        out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2] * self.subsample[0], img.shape[3] * self.subsample[1])
        self.output = GpuDnnConvGradI()(kerns, img, out, desc) + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.output

    def get_output_train(self, input):
        self.output_train = self.get_output(input)
        return self.output_train


class Pool(object):
    def __init__(self, poolsize=(2, 2)):
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


class Maxout(object):
    def __init__(self, n_out, size=4, W_values=None, b_values=None, init='glorot_uniform'):
        self.n_in = None
        self.n_out = n_out
        self.size = size
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
        self.fan_in = self.n_in
        self.fan_out = self.n_out*self.size
        self.W_values = np.asarray(self.init(self, (self.n_in, self.n_out*self.size)).reshape(self.size, self.n_in, self.n_out),
                                   dtype=theano.config.floatX)

        self.W = theano.shared(value=self.W_values, name='W', borrow=True)
        if self.b_values is None:
            self.b_values = np.zeros((self.size, self.n_out), dtype=theano.config.floatX)
        self.b = theano.shared(value=self.b_values, name='b', borrow=True)

        self.params = [self.W, self.b]

    def get_output(self, input):
        self.output = T.max(T.dot(input, self.W) + self.b, axis=1)
        return self.output

    def get_output_train(self, input):
        self.output_train = self.get_output(input)
        return self.output_train


class Flatten(object):
    def __init__(self):
        self.n_in = None
        self.n_out = None
        self.output = None
        self.output_train = None

    def set_input_shape(self, n_in):
        self.n_in = n_in
        self.n_out = self.n_in[0] * self.n_in[1] * self.n_in[2]

    def get_output(self, input):
        self.output = input.flatten(2)
        return self.output

    def get_output_train(self, input):
        self.output_train = self.get_output(input)
        return self.output_train


class GaussianNoise(object):
    def __init__(self, std=0.1):
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


class Decoder(object):
    def __init__(self, encoder, b_values=None):
        self.n_in = encoder.n_out
        self.n_out = encoder.n_in
        self.W = encoder.W.T
        self.b_values = b_values
        self.b = None
        self.params = None
        self.output = None
        self.output_train = None

    def set_input_shape(self, n_in):
        if n_in != self.n_in:
            raise ValueError("Definition of Decoder is wrong.")

    def set_params(self):
        if self.b_values is None:
            self.b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
        else:
            if not isinstance(self.b_values, np.ndarray):
                raise TypeError("type(b_values) must be numpy.ndarray.")
            if self.b.shape != (self.n_out,):
                raise ValueError("b_values.shape must be (n_out,)")
            if self.b_values.dtype != theano.config.flaotX:
                raise ValueError("b_values.dtype must be theano.config.flaotX.")
        self.b = theano.shared(value=self.b_values, name='b', borrow=True)

        self.params = [self.b]

    def get_output(self, input):
        self.output = T.dot(input, self.W) + self.b
        return self.output

    def get_output_train(self, input):
        self.output_train = self.get_output(input)
        return self.output_train


class Reshape(object):
    def __init__(self, shape):
        self.n_in = None
        self.n_out = shape
        self.output = None

    def set_input_shape(self, n_in):
        self.n_in = n_in

    def get_output(self, input):
        self.output = T.reshape(input, (input.shape[0], self.n_out[0], self.n_out[1], self.n_out[2]))
        return self.output

    def get_output_train(self, input):
        return self.get_output(input)
