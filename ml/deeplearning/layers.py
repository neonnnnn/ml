
from abc import ABCMeta, abstractmethod
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


class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_shape(self, n_in):
        pass

    @abstractmethod
    def forward(self, x, train=True):
        pass

    def __call__(self, x, train=True):
        return self.forward(x, train)


class Dense(Layer):
    def __init__(self, n_out, n_in=None, init='glorot_uniform',
                 rng=None, bias=True):
        self.n_in = n_in
        self.n_out = n_out
        self.rng = rng
        self.W = None
        self.b = None
        self.bias = bias
        self.init = initializations.get_init(init)
        self.params = None

    def set_rng(self, rng):
        self.rng = rng

    def set_shape(self, n_in):
        self.n_in = n_in
        if self.W is not None:
            if self.W.get_value().shape != (self.n_in, self.n_out):
                raise ValueError('W_values.shape must be '
                                 '(n_in, n_out)(i.e. {0})'
                                 .format((self.n_in, self.n_out)))

    def set_params(self):
        if self.W is None:
            W_values = np.asarray(self.init(self, (self.n_in, self.n_out)),
                                  dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, name='W', borrow=True)
        if not self.bias:
            self.params = [self.W]
        else:
            if self.b is None:
                b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
                self.b = theano.shared(value=b_values, name='b', borrow=True)
            self.params = [self.W, self.b]

    def set_weight(self, W_values):
        if not isinstance(W_values, np.ndarray):
            raise TypeError('type(W_values) must be numpy.ndarray.')

        if W_values.dtype != theano.config.floatX:
            raise ValueError('W_values.dtype must be {0}'
                             .format(theano.config.floatX))

        if self.n_in is not None:
            if W_values.shape != (self.n_in, self.n_out):
                raise ValueError('W_values.shape must be '
                                 '(n_in, n_out)(i.e. {0})'
                                 .format((self.n_in, self.n_out)))

        if self.W is not None:
            self.W.set_value(W_values)
        else:
            self.W = theano.shared(value=W_values, name='W', borrow=True)

    def set_bias(self, b_values):
        if not isinstance(b_values, np.ndarray):
            raise TypeError('type(b_values')

        if b_values.dtype != theano.config.floatX:
            raise ValueError('W_values.dtype must be {0}'
                             .format(theano.config.floatX))

        if b_values.shape != (self.n_out, ):
            raise ValueError('b_values.shape must be (n_out, ), i.e. {0}.'
                             .format((self.n_out,)))

        if self.b is not None:
            self.b.set_value(b_values)
        else:
            self.b = theano.shared(value=b_values, borrow=True)

    def forward(self, x, train=True):
        return T.dot(x, self.W) + self.b


class Activation(Layer):
    def __init__(self, activation_name, *args):
        self.n_in = None
        self.n_out = None
        self.afunc = activations.get_activation(activation_name)
        self.act_param = args

    def set_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def forward(self, x, train=True):
        if inspect.getargspec(self.afunc)[0] and self.act_param:
            return self.afunc(x, self.act_param)
        else:
            return self.afunc(x)


class BatchNormalization(Layer):
    def __init__(self, n_in=None, eps=1e-5, trainable=True, momentum=0.9,
                 moving=True):
        self.n_in = n_in
        self.n_out = n_in
        self.mean_inf = None
        self.var_inf = None
        self.gamma = 1.0
        self.beta = 0.0
        self.params = None
        self.eps = theano.shared(np.asarray(eps, dtype=theano.config.floatX),
                                 borrow=True)
        self.momentum = momentum
        self.trainable = trainable
        self.moving = moving
        self.updates = None

    def set_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def set_params(self):
        if isinstance(self.n_in, int):
            shape = self.n_in
        else:
            shape = self.n_in[0]
        if self.moving:
            self.mean_inf = theano.shared(np.zeros(shape,
                                                   dtype=theano.config.floatX),
                                          borrow=True)
            self.var_inf = theano.shared(np.zeros(shape,
                                                  dtype=theano.config.floatX),
                                         borrow=True)

        self.gamma = theano.shared(np.ones(shape, dtype=theano.config.floatX),
                                   borrow=True)
        self.beta = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                  borrow=True)

        if self.trainable:
            self.params = [self.gamma, self.beta]
        else:
            self.params = []

        self.momentum = theano.shared(np.asarray(self.momentum,
                                                 dtype=theano.config.floatX),
                                      borrow=True)

    def forward(self, x, train=True):
        if train or not self.moving:
            if x.ndim == 2:
                mean = T.mean(x, axis=0)
                var = T.var(x, axis=0)
                output = self.gamma * (x - mean) / T.sqrt(var + self.eps)
                output += self.beta
            elif x.ndim == 4:
                gamma = self.gamma.dimshuffle('x', 0, 'x', 'x')
                beta = self.beta.dimshuffle('x', 0, 'x', 'x')
                mean = T.mean(x, axis=(0, 2, 3))
                var = T.var(x, axis=(0, 2, 3))
                output = gamma*(x - mean.reshape((1, x.shape[1], 1, 1)))
                output /= T.sqrt(var).reshape((1, x.shape[1], 1, 1)) + self.eps
                output += beta
            else:
                raise ValueError('input.shape must be (batch_size, dim) '
                                 'or (batch_size, filter_num, h, w).')

            if self.moving:
                bs = x.shape[0].astype(theano.config.floatX)
                self.updates = \
                    [(self.mean_inf,
                      self.momentum*self.mean_inf + (1-self.momentum)*mean),
                     (self.var_inf,
                      (self.momentum*self.var_inf
                       + (1-self.momentum)*var*bs/(bs - 1.)))]
            else:
                self.updates = []
        else:
            if x.ndim == 2:
                output = self.gamma * (x - self.mean_inf)
                output /= T.sqrt(self.var_inf) +self.eps
                output += self.beta
            elif x.ndim == 4:
                gamma = self.gamma.dimshuffle('x', 0, 'x', 'x')
                mean_inf = self.mean_inf.dimshuffle('x', 0, 'x', 'x')
                var_inf = self.var_inf.dimshuffle('x', 0, 'x', 'x')
                beta = self.beta.dimshuffle('x', 0, 'x', 'x')
                output = gamma*(x-mean_inf)/(T.sqrt(var_inf)+self.eps) + beta
            else:
                raise ValueError('input.shape must be (batch_size, dim) '
                                 'or (batch_size, filter_num, h, w).')
        return output


class Dropout(Layer):
    def __init__(self, p=0.5, n_in=None, rng=None):
        self.n_in = n_in
        self.n_out = n_in
        self.rng = rng
        self.p = p

    def set_rng(self, rng):
        self.rng = rng

    def set_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def forward(self, x, train=True):
        if train:
            srng = T.shared_randomstreams.RandomStreams(self.rng.randint(9999))
            output = x * srng.binomial(size=x.shape, n=1,
                                       p=1 - self.p, dtype=x.dtype)
        else:
            output = x * (1 - self.p)
        return output


class Conv(Layer):
    def __init__(self, nb_filter, nb_height, nb_width, n_in=None, n_out=None,
                 border_mode='valid', init='he_conv_normal',
                 subsample=(1, 1), rng=None):
        self.n_in = n_in
        self.n_out = n_out
        self.rng = rng
        self.filter_shape = [nb_filter, None, nb_height, nb_width]
        self.W = None
        self.b = None
        self.params = None
        self.init = initializations.get_init(init)
        self.border_mode = border_mode
        self.subsample = subsample

    def set_rng(self, rng):
        self.rng = rng

    def set_shape(self, n_in):
        self.n_in = n_in
        self.filter_shape[1] = self.n_in[0]
        self.filter_shape = tuple(self.filter_shape)
        if self.border_mode == 'valid':
            pad = [0, 0]
        elif self.border_mode == 'full':
            pad = [self.filter_shape[2] - 1, self.filter_shape[3] - 1]
        elif self.border_mode == 'half':
            pad = [self.filter_shape[2] // 2, self.filter_shape[3] // 2]
        elif isinstance(self.border_mode, (list, tuple)):
            pad = self.border_mode
        elif isinstance(self.border_mode, int):
            if self.border_mode < 0:
                raise ValueError("border_mode must be >= 0.")
            pad = [self.border_mode, self.border_mode]
        else:
            raise ValueError('invalid border_mode {}, which must be either '
                             '"valid", "full", "half", an integer '
                             'or a pair of integers'.format(self.border_mode))
        fmap_h = ((self.n_in[1] - self.filter_shape[2] + 2 * pad[0])
                  // self.subsample[0] + 1)
        fmap_w = ((self.n_in[2] - self.filter_shape[3] + 2 * pad[1])
                  // self.subsample[1] + 1)

        if fmap_h <= 0 or fmap_w <= 0:
            raise ValueError('Output height or width must be > 0, '
                             'but now height = {0}, width={1}.'
                             .format(fmap_h, fmap_w))

        self.n_out = (self.filter_shape[0], fmap_h, fmap_w)

    def set_params(self):
        if self.W is None:
            W_values = np.asarray(self.init(self, self.filter_shape),
                                  dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, name='W', borrow=True)

        if self.b is None:
            b_values = np.zeros((self.filter_shape[0],),
                                dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.params = [self.W, self.b]

    def set_weight(self, W_values):
        if not isinstance(W_values, np.ndarray):
            raise TypeError('type(W_values) must be numpy.ndarray.')

        if W_values.dtype != theano.config.floatX:
            raise ValueError('W_values.dtype must be {0}'
                             .format(theano.config.floatX))

        if self.n_in is not None:
            if W_values.shape != self.filter_shape:
                raise ValueError('W_values.shape must be (nb_filter, n_in[0], '
                                 'nb_height, nb_width), i.e. {0}.'
                                 .format(self.filter_shape))

        if self.W is not None:
            self.W.set_value(W_values)
        else:
            self.W = theano.shared(value=W_values, name='W', borrow=True)

    def set_bias(self, b_values):
        if not isinstance(b_values, np.ndarray):
            raise TypeError('type(b_values) must be numpy.ndarry.')

        if b_values.shape != self.filter_shape[0]:
            raise ValueError('b_balues.shape must be (nb_filter,), i.e. {0}.'
                             .format((self.filter_shape[0],)))

        if b_values.dtype != theano.config.floatX:
            raise ValueError('b_values.dtype must be {0}.'
                             .format(theano.config.floatX))

        if self.b is not None:
            self.b.set_value(b_values)
        else:
            self.b = theano.shared(value=b_values, name='b', borrow=True)

    def forward(self, x, train=True):
        conv_out = conv2d(
            input=x,
            filters=self.W,
            filter_shape=self.filter_shape,
            border_mode=self.border_mode,
            subsample=self.subsample
        )
        return conv_out + self.b.dimshuffle('x', 0, 'x', 'x')


class Deconv(Conv):
    def __init__(self, nb_filter, nb_height, nb_width, n_out, n_in=None,
                 border_mode='adapt', init='he_conv_normal',
                 subsample=(1, 1), rng=None):
        super(Deconv, self).__init__(nb_filter, nb_height, nb_width, n_in,
                                     n_out, border_mode, init, subsample, rng)

    def set_rng(self, rng):
        self.rng = rng

    def set_shape(self, n_in):
        self.n_in = n_in
        self.filter_shape[1] = self.n_in[0]
        self.filter_shape = tuple(self.filter_shape)

        if self.n_out[0] != self.filter_shape[0]:
            raise ValueError('output_shape[0] must be equal nb_filter.')

        if self.border_mode == 'valid':
            pad = [0, 0]
        elif self.border_mode == 'full':
            pad = [self.filter_shape[2] - 1, self.filter_shape[3] - 1]
        elif self.border_mode == 'half':
            pad = [self.filter_shape[2] // 2, self.filter_shape[3] // 2]
        elif isinstance(self.border_mode, (list, tuple)):
            pad = self.border_mode
        elif isinstance(self.border_mode, int):
            if self.border_mode < 0:
                raise ValueError('border_mode must be >= 0.')
            pad = [self.border_mode, self.border_mode]
        elif self.border_mode == 'adapt':
            pad = [0, 0]
            inf = np.array(self.subsample) * (np.array(n_in[1:]) - 1) \
                  + np.array(self.filter_shape[2:])
            sup = inf + np.array(self.subsample) - 1
            if self.n_out[1] > sup[0]:
                raise ValueError('border_mode is "adapt" but output_shape[1] '
                                 'is impossible. filter_size or subsample '
                                 'must be larger.')
            elif self.n_out[2] > sup[1]:
                raise ValueError('border_mode is "adapt" but output_shape[2] '
                                 'is impossible. filter_size or subsample '
                                 'must be larger.')
            elif self.n_out[1] < inf[0] & self.n_out[2] < inf[1]:
                pad[0] = (inf[0] - self.n_out[1]) // 2
                pad[1] = (inf[1] - self.n_out[2]) // 2
                self.border_mode = pad
        else:
            raise ValueError('invalid border_mode {}, '
                             'which must be either "valid", "full", "half", '
                             'an integer or a pair of integers'
                             .format(self.border_mode))

        inf = np.array(self.subsample) * (np.array(n_in[1:]) - 1) \
              + np.array(self.filter_shape[2:]) - 2 * np.array(pad)
        sup = inf + np.array(self.subsample) - 1
        if not inf[0] <= self.n_out[1] <= sup[0]:
            if inf[1] <= self.n_out[2] <= sup[1]:
                raise ValueError('impossible output_shape. output = '
                                 'subsample * (x - 1) + filter - 2 * pad + a, '
                                 'a \\in {0, \\ldots , subsample-1}')

    def set_params(self):
        if self.W is None:
            W_values = np.asarray(self.init(self, self.filter_shape),
                                  dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, borrow=True)

        if self.b is None:
            b_values = np.zeros((self.filter_shape[0],),
                                dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)

        self.params = [self.W, self.b]

    def forward(self, x, train=True):
        op = T.nnet.abstract_conv.AbstractConv2d_gradxs(
            imshp=(None, self.n_out[0], self.n_out[1], self.n_out[2]),
            kshp=self.filter_shape,
            subsample=self.subsample,
            border_mode=self.border_mode,
            filter_flip=True
        )
        deconv_out = op(self.W.dimshuffle(1, 0, 2, 3), x, self.n_out[1:])
        return deconv_out + self.b.dimshuffle('x', 0, 'x', 'x')
    

class ConvCUDNN(Conv):
    def forward(self, x, train=True):
        conv_out = dnn_conv(
            img=x,
            kerns=self.W,
            border_mode=self.border_mode,
            subsample=self.subsample
        )
        return conv_out + self.b.dimshuffle('x', 0, 'x', 'x')


class DeconvCUDNN(Deconv):
    def forward(self, x, train=True):
        img = gpu_contiguous(x)
        kerns = gpu_contiguous(self.W.dimshuffle(1, 0, 2, 3))
        gpudnnconvdesc = GpuDnnConvDesc(border_mode=self.border_mode,
                                        subsample=self.subsample,
                                        conv_mode="conv")
        out = gpu_alloc_empty(img.shape[0],
                              kerns.shape[1],
                              img.shape[2] * self.subsample[0],
                              img.shape[3] * self.subsample[1])
        desc = gpudnnconvdesc(out.shape, kerns.shape)
        return (GpuDnnConvGradI()(kerns, img, out, desc)
                + self.b.dimshuffle('x', 0, 'x', 'x'))


class Pool(Layer):
    def __init__(self, poolsize=(2, 2)):
        self.n_in = None
        self.n_out = None
        self.poolsize = poolsize

    def set_shape(self, n_in):
        self.n_in = n_in
        self.n_out = (self.n_in[0],
                      self.n_in[1] / self.poolsize[0],
                      self.n_in[2] / self.poolsize[1])

    def forward(self, x, train=True):
        output = pool.pool_2d(
                input=x,
                ds=self.poolsize,
                ignore_border=True
        )
        return output


class Flatten(Layer):
    def __init__(self):
        self.n_in = None
        self.n_out = None

    def set_shape(self, n_in):
        self.n_in = n_in
        self.n_out = self.n_in[0] * self.n_in[1] * self.n_in[2]

    def forward(self, x, train=True):
        return x.flatten(2)


class GaussianNoise(Layer):
    def __init__(self, std=0.1, rng=None):
        self.n_in = None
        self.n_out = None
        self.rng = rng
        self.std = std

    def set_rng(self, rng):
        self.rng = rng

    def set_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def forward(self, x, train=True):
        if train:
            srng = T.shared_randomstreams.RandomStreams(self.rng.randint(9999))
            output = x + srng.normal(size=x.shape, avg=0, std=self.std,
                                     dtype=x.dtype)
        else:
            output = x
        return output


class Decoder(Layer):
    def __init__(self, encoder):
        self.n_in = encoder.n_out
        self.n_out = encoder.n_in
        self.W = encoder.W.T
        self.b = None
        self.params = None

    def set_shape(self, n_in):
        if n_in != self.n_in:
            raise ValueError("Definition of Decoder is wrong.")

    def set_params(self):
        if self.b is None:
            b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.params = [self.b]

    def forward(self, x, train=True):
        return T.dot(x, self.W) + self.b


class Reshape(Layer):
    def __init__(self, shape):
        self.n_in = None
        self.n_out = tuple(shape)

    def set_shape(self, n_in):
        self.n_in = n_in

    def forward(self, x, train=True):
        return T.reshape(x, (x.shape[0], ) + self.n_out)
