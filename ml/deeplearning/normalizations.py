from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from .layers import Layer
import theano
import theano.tensor as T
from .theanoutils import sharedasarray, sharedones, sharedzeros


class Normalization(Layer):
    __metaclass__ = ABCMeta

    def __init__(self, layer):
        self.layer = layer
        self.n_in = self.layer.n_in
        self.n_out = self.layer.n_out

    def set_shape(self, n_in):
        self.layer.set_shape(n_in)
        self.n_in = self.layer.n_in
        self.n_out = self.layer.n_out

    def set_rng(self, rng):
        self.layer.set_rng(rng)

    @abstractmethod
    def forward(self, x, train=True):
        pass

    def __call__(self, x, train=True):
        return self.forward(x, train)

    def get_layers_with_names_configs(self):
        return [self.layer], 'layer', {'layer': self.layer.get_config()}


class BatchNormalization(Normalization):
    def __init__(self, layer, eps=1e-8, trainable=True, momentum=0.99, moving=True):
        super(BatchNormalization, self).__init__(layer)
        self.mean_inf = None
        self.var_inf = None
        self.gamma = None
        self.beta = None
        self.params = None
        self.eps = sharedasarray(eps)
        self.momentum = momentum
        self.trainable = trainable
        self.moving = moving
        self.updates = None

    def get_updates(self):
        return self.updates

    def set_params(self):
        self.layer.set_params()
        if isinstance(self.n_out, int):
            shape = self.n_out
        else:
            shape = self.n_out[0]
        if self.moving and self.mean_inf is None and self.var_inf is None:
            self.mean_inf = sharedzeros(shape)
            self.var_inf = sharedzeros(shape)

        if self.trainable:
            if self.gamma is None:
                self.gamma = sharedones(shape)
            if self.beta is None:
                self.beta = sharedzeros(shape)
            self.params = [self.gamma, self.beta]
        else:
            self.params = []
        if isinstance(self.momentum, float):
            self.momentum = sharedasarray(self.momentum)

        self.params += self.layer.params

    def forward(self, x, train=True):
        x = self.layer.forward(x, train)
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


class WeighNormalization(Normalization):
    def __init__(self, layer, trainable=True):
        self.g = None
        self.params = None
        self.trainable = trainable
        super(WeighNormalization, self).__init__(layer)

    def set_params(self):
        self.layer.set_params()
        if isinstance(self.n_in, int):
            shape = (self.n_in, 1)
        else:
            shape = (self.n_in[0], 1, 1, 1)

        if self.trainable:
            if self.g is None:
                self.g = sharedones(shape)
            self.params = [self.g]
        else:
            self.params = []

        if self.layer.W.ndim == 2:
            axis = 0
        else:
            axis = (1, 2, 3)

        W_norm = T.sqrt(T.sum(self.layer.W ** 2, axis=axis, keepdims=True))

        if self.g is not None:
            self.layer.W *= (self.g / W_norm)
        else:
            self.layer.W /= W_norm

        self.params += self.layer.params

    def forward(self, x, train=True):
        return self.layer.forward(x, train)
