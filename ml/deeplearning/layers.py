from abc import ABCMeta, abstractmethod
import numpy as np
import theano
import theano.tensor as T
from theanoutils import sharedasarray, sharedones, sharedzeros
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

    def get_config(self):
        config = {'class': self.__class__.__name__,
                  'n_in': self.n_in,
                  'n_out': self.n_out,
                  }
        return config


class Dense(Layer):
    def __init__(self, n_out,
                 n_in=None,
                 init='normal',
                 rng=None,
                 bias=True,
                 W=None,
                 b=None):
        self.n_in = n_in
        self.n_out = n_out
        self.rng = rng
        self.W = sharedasarray(W, 'W')
        self.b = sharedasarray(b, 'b')
        self.bias = bias
        if isinstance(init, str):
            self.init = initializations.get_init(init)()
        else:
            self.init = init
        self.params = None

    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({'bias': self.bias})
        return config

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
            self.W = sharedasarray(self.init(self, (self.n_in, self.n_out)),
                                   'W')
        if not self.bias:
            self.params = [self.W]
        else:
            if self.b is None:
                self.b = sharedzeros((self.n_out,), 'b')
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
            self.W = theano.shared(W_values, name='W', borrow=True)

    def set_bias(self, b_values):
        if not isinstance(b_values, np.ndarray):
            raise TypeError('type(b_values) must be numpy.ndarray')

        if b_values.dtype != theano.config.floatX:
            raise ValueError('b_values.dtype must be {0}'
                             .format(theano.config.floatX))

        if b_values.shape != (self.n_out, ):
            raise ValueError('b_values.shape must be (n_out, ), i.e., {0}.'
                             .format((self.n_out,)))

        if self.b is not None:
            self.b.set_value(b_values)
        else:
            self.b = theano.shared(value=b_values, name='b', borrow=True)

    def forward(self, x, train=True):
        return T.dot(x, self.W) + self.b


class Activation(Layer):
    def __init__(self, activation_name, *args):
        self.n_in = None
        self.n_out = None
        self.afunc = activations.get_activation(activation_name)
        self.act_param = args

    def get_config(self):
        config = super(Activation, self).get_config()
        config.update({'act': self.afunc.__name__})
        return config

    def set_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def forward(self, x, train=True):
        if inspect.getargspec(self.afunc)[0] and self.act_param:
            return self.afunc(x, self.act_param)
        else:
            return self.afunc(x)


class BatchNormalization(Layer):
    def __init__(self,
                 n_in=None,
                 eps=1e-5,
                 trainable=True,
                 momentum=0.95,
                 moving=True,
                 mean_inf=None,
                 var_inf=None,
                 beta=None,
                 gamma=None):
        self.n_in = n_in
        self.n_out = n_in
        self.mean_inf = sharedasarray(mean_inf, 'mean_inf')
        self.var_inf = sharedasarray(var_inf, 'var_inf')
        self.gamma = sharedasarray(gamma, 'gamma')
        self.beta = sharedasarray(beta, 'beta')
        self.params = None
        self.eps = sharedasarray(eps, 'eps')
        self.momentum = momentum
        self.trainable = trainable
        self.moving = moving
        self.updates = None

    def set_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def get_updates(self):
        return self.updates

    def set_params(self):
        if isinstance(self.n_in, int):
            shape = self.n_in
        else:
            shape = self.n_in[0]
        if self.moving and self.mean_inf is None and self.var_inf is None:
            self.mean_inf = sharedzeros(shape, 'mean_inf')
            self.var_inf = sharedzeros(shape, 'var_inf')

        if self.trainable:
            if self.gamma is None:
                self.gamma = sharedones(shape, 'gamma')
            if self.beta is None:
                self.beta = sharedzeros(shape, 'beta')
            self.params = [self.gamma, self.beta]
        else:
            self.params = []
        if isinstance(self.momentum, float):
            self.momentum = sharedasarray(self.momentum, 'momentum')

    def forward(self, x, train=True):
        if train or (not self.moving):
            if x.ndim == 2:
                mean = T.mean(x, axis=0)
                var = T.var(x, axis=0)
            elif x.ndim == 4:
                mean = T.mean(x, axis=(0, 2, 3))
                var = T.var(x, axis=(0, 2, 3))
            else:
                raise ValueError('input.shape must be (batch_size, dim) '
                                 'or (batch_size, filter_num, h, w).')
            if self.moving:
                bs = x.shape[0].astype(theano.config.floatX)
                mean_inf_next = (self.momentum*self.mean_inf +
                                 (1-self.momentum)*mean)
                var_inf_next = (self.momentum*self.var_inf
                                + (1-self.momentum)*var*bs/(bs-1.))
                self.updates = [(self.mean_inf, mean_inf_next),
                                (self.var_inf, var_inf_next)]
            else:
                self.updates = []
        else:
            mean = self.mean_inf
            var = self.var_inf

        if x.ndim == 4:
            mean = mean.dimshuffle('x', 0, 'x', 'x')
            var = var.dimshuffle('x', 0, 'x', 'x')

        output = (x-mean) / T.sqrt(var+self.eps)

        if self.gamma is not None:
            if x.ndim == 4:
                output *= self.gamma.dimshuffle('x', 0, 'x', 'x')
            else:
                output *= self.gamma
        if self.beta is not None:
            if x.ndim == 4:
                output += self.beta.dimshuffle('x', 0, 'x', 'x')
            else:
                output += self.beta

        return output


class Dropout(Layer):
    def __init__(self, p=0.5, n_in=None, rng=None):
        self.n_in = n_in
        self.n_out = n_in
        self.rng = rng
        self.p = p
        self.srng = None

    def set_rng(self, rng):
        self.rng = rng
        self.srng = T.shared_randomstreams.RandomStreams(rng.randint(9999))

    def set_shape(self, n_in):
        self.n_in = n_in
        self.n_out = n_in

    def forward(self, x, train=True):
        if train:
            output = x * self.srng.binomial(size=x.shape, n=1,
                                            p=1-self.p, dtype=x.dtype)
        else:
            output = x * (1-self.p)
        return output


class Conv(Layer):
    def __init__(self,
                 nb_filter,
                 nb_height,
                 nb_width,
                 n_in=None,
                 n_out=None,
                 border_mode='valid',
                 init='he_conv_normal',
                 subsample=(1, 1),
                 rng=None,
                 W=None,
                 b=None):
        self.nb_filter = None
        self.nb_heigt = nb_height
        self.nb_weight = nb_width
        self.n_in = n_in
        self.n_out = n_out
        self.rng = rng
        self.filter_shape = [nb_filter, None, nb_height, nb_width]
        self.W = sharedasarray(W, 'W')
        self.b = sharedasarray(b, 'b')
        self.params = None
        if isinstance(init, str):
            self.init = initializations.get_init(init)()
        else:
            self.init = init
        self.border_mode = border_mode
        self.subsample = subsample

    def set_rng(self, rng):
        self.rng = rng

    def set_shape(self, n_in):
        self.n_in = n_in
        if self.filter_shape[1] is None:
            self.filter_shape[1] = self.n_in[0]
            self.filter_shape = tuple(self.filter_shape)
        if self.border_mode == 'valid':
            pad = [0, 0]
        elif self.border_mode == 'full':
            pad = [self.filter_shape[2]-1, self.filter_shape[3]-1]
        elif self.border_mode == 'half':
            pad = [self.filter_shape[2]//2, self.filter_shape[3]//2]
        elif isinstance(self.border_mode, (list, tuple)):
            pad = self.border_mode
        elif isinstance(self.border_mode, int):
            if self.border_mode < 0:
                raise ValueError('border_mode must be >= 0.')
            pad = [self.border_mode, self.border_mode]
        else:
            raise ValueError('invalid border_mode {}, which must be either '
                             '"valid", "full", "half", an integer '
                             'or a pair of integers'.format(self.border_mode))
        fmap_h = ((self.n_in[1]-self.filter_shape[2]+2*pad[0])
                  // self.subsample[0] + 1)
        fmap_w = ((self.n_in[2]-self.filter_shape[3]+2*pad[1])
                  // self.subsample[1] + 1)

        if fmap_h <= 0 or fmap_w <= 0:
            raise ValueError('Output height or width must be > 0, '
                             'but now height = {0}, width={1}.'
                             .format(fmap_h, fmap_w))

        self.n_out = (self.filter_shape[0], fmap_h, fmap_w)

    def set_params(self):
        if self.W is None:
            self.W = sharedasarray(self.init(self, self.filter_shape), 'W')

        if self.b is None:
            self.b = sharedzeros((self.filter_shape[0],), 'b')

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
            raise TypeError('type(b_values) must be numpy.ndarray.')

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
    def __init__(self,
                 nb_filter,
                 nb_height,
                 nb_width,
                 n_out,
                 n_in=None,
                 border_mode='adapt',
                 init='he_conv_normal',
                 subsample=(1, 1),
                 rng=None,
                 W=None,
                 b=None):
        super(Deconv, self).__init__(nb_filter,
                                     nb_height,
                                     nb_width,
                                     n_in,
                                     n_out,
                                     border_mode,
                                     init,
                                     subsample,
                                     rng,
                                     W,
                                     b)

    def set_rng(self, rng):
        self.rng = rng

    def set_shape(self, n_in):
        self.n_in = n_in
        if self.filter_shape[1] is None:
            self.filter_shape[1] = self.n_in[0]
            self.filter_shape = self.filter_shape

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
            inf = (np.array(self.subsample)*(np.array(n_in[1:]) - 1)
                   + np.array(self.filter_shape[2:]))
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
                pad[0] = (inf[0]-self.n_out[1]) // 2
                pad[1] = (inf[1]-self.n_out[2]) // 2
                self.border_mode = pad
        else:
            raise ValueError('invalid border_mode {}, '
                             'which must be either "valid", "full", "half", '
                             'an integer or a pair of integers'
                             .format(self.border_mode))

        inf = (np.array(self.subsample)*(np.array(n_in[1:]) - 1)
               + np.array(self.filter_shape[2:])-2*np.array(pad))
        sup = inf + np.array(self.subsample) - 1
        if not inf[0] <= self.n_out[1] <= sup[0]:
            if inf[1] <= self.n_out[2] <= sup[1]:
                raise ValueError('impossible output_shape. output = '
                                 'subsample * (x - 1) + filter - 2 * pad + a, '
                                 'a \\in {0, \\ldots , subsample-1}')

    def set_params(self):
        if self.W is None:
            self.W = sharedasarray(self.init(self, self.filter_shape), 'W')

        if self.b is None:
            self.b = sharedzeros((self.filter_shape[0],), 'b')

        self.params = [self.W, self.b]

    def forward(self, x, train=True):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
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
                                        conv_mode='conv')
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
        output = pool.pool_2d(input=x, ds=self.poolsize, ignore_border=True)
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


class Concat(Layer):
    # concatenation of 1d outputs
    def __init__(self, layers=None, axis=1):
        self.layers = layers
        self.n_out = 0
        self.n_in = []
        self.axis = axis
        if self.layers is not None:
            for layer in self.layers:
                self.n_in.append(layer.n_out)

    def set_shape(self, n_in):
        if self.n_in != n_in:
            raise ValueError('The values of n_in is wrong.')
        if isinstance(self.n_in[0], int):
            if self.axis == 1:
                self.n_out = sum(self.n_in)
            else:
                raise ValueError('self.axis must be 1 when concatenating 1D (Dense) layers.')
        else:
            n_out = list(self.n_in[0])
            for n_in in self.n_in[1:]:
                if len(n_out) != n_in:
                    raise ValueError('len(n_in) of all concatenated layers  must be equal.')

        self.n_out = tuple(n_out)

    def forward(self, tensors=None, train=True):
        if self.layers is not None:
            return T.concatenate(tensors, axis=self.axis)
        else:
            concat = [self.layer.forward(tensor, train)
                      for layer, tensor in zip(self.layers, tensors)]
            return T.concatenate(concat, axis=self.axis)


class Decoder(Layer):
    def __init__(self, encoder):
        self.n_in = encoder.n_out
        self.n_out = encoder.n_in
        self.W = encoder.W.T
        self.b = None
        self.params = None

    def set_shape(self, n_in):
        if n_in != self.n_in:
            raise ValueError('Definition of Decoder is wrong.')

    def set_params(self):
        if self.b is None:
            self.b = sharedzeros((self.n_out,), 'b')

        self.params = [self.b]

    def forward(self, x, train=True):
        return T.dot(x, self.W) + self.b


class Reshape(Layer):
    def __init__(self, n_out, layer=None):
        self.layer = layer
        self.n_in = None
        self.n_out = tuple(n_out)
        self.params = None

    def set_rng(self, rng):
        if self.layer is not None:
            self.layer.set_rng(rng)

    def set_shape(self, n_in):
        if self.layer is not None:
            self.layer.set_shape(n_in)
            self.n_in = self.layer.n_out
        else:
            self.n_in = n_in

    def set_params(self):
        if hasattr(self.layer, 'set_params'):
            self.layer.set_params()
            self.params = self.layer.params
        else:
            self.params = []

    def get_updates(self):
        if hasattr(self.layer, 'get_updates'):
            return self.layer.get_updates()
        else:
            return []

    def forward(self, x, train=True):
        if self.layer is not None:
            f = self.layer.forward(x, train)
        else:
            f = x
        return T.reshape(f, (f.shape[0], )+self.n_out)
