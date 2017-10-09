from __future__ import absolute_import

import theano.tensor as T
from .. import utils
from abc import ABCMeta, abstractmethod


class Activation(object):
    __metaclass__ = ABCMeta

    def __init__(self, layer):
        self.layer = layer
        self.n_out = layer.n_out
        self.n_in = layer.n_in
        if hasattr(self.layer, 'rng'):
            setattr(self, 'None', None)
        if hasattr(self.layer, 'params'):
            setattr(self, 'params', None)
        if hasattr(self.layer, 'updates'):
            setattr(self, 'updates', None)

    @abstractmethod
    def __call__(self, x):
        pass

    def forward(self, x, train=True):
        self.__call__(self.layer.forward(x, train))

    def set_rng(self, rng):
        if hasattr(self.layer, 'rng'):
            self.layer.set_rng(rng)

    def set_params(self):
        self.layer.set_params()
        setattr(self, 'params', self.layer.params)


class Sigmoid(Activation):
    def __init__(self, layer=None, alpha=1.0):
        super(Sigmoid, self).__init__(layer)
        self.alpha = alpha

    def __call__(self, x):
        return T.nnet.sigmoid(self.alpha*x)


def sigmoid(x, alpha=1.0):
    return T.nnet.sigmoid(alpha*x)


class Tanh(Activation):
    def __init__(self, layer=None, alpha=1.0):
        super(Tanh, self).__init__(layer)
        self.alpha = alpha

    def __call__(self, x):
        return T.tanh(self.alpha*x)


def tanh(x, alpha=1.0):
    return T.tanh(alpha*x)


class Softmax(Activation):
    def __call__(self, x):
        return softmax(x)


def softmax(x):
    if x.ndim == 3:
        true_shape = x.shape
        reshape = (true_shape[0]*true_shape[1], true_shape[2])
        x_reshape = x.reshape(reshape) - T.max(x.reshape(reshape), axis=1,
                                               keepdims=True)
        activation = T.nnet.softmax(x_reshape)
        activation = activation.reshape(true_shape)
    else:
        activation = T.nnet.softmax(x)

    return activation


class ReLU(Activation):
    def __call__(self, x):
        return T.nnet.relu(x)


def relu(x):
    return T.nnet.relu(x)


class Softplus(Activation):
    def __call__(self, x):
        return T.nnet.softplus(x)


def softplus(x):
    return T.maximum(0, x) + T.log(1+T.exp(-T.abs_(x)))


class LeakyReLU(Activation):
    def __init__(self, layer=None, alpha=0.2):
        super(LeakyReLU, self).__init__(layer)
        if alpha <= 0 or alpha >= 1:
            raise ValueError('0 < alpha < 1.')
        self.alpha = alpha

    def __call__(self, x):
        return T.nnet.relu(x, self.alpha)


def leakyrelu(x, alpha=0.2):
    return T.nnet.relu(x, alpha)


class Maxout(Activation):
    def __init__(self, layer=None, pool_size=4):
        super(Maxout, self).__init__(layer)
        self.n_out = self.layer.n_out / pool_size
        if pool_size <= 0:
            raise ValueError('pool_size must be Natural number.')
        self.pool_size = pool_size

    def __call__(self, x):
        if x.shape[1] % self.pool_size == 0:
            raise ValueError('x.shape[1] must be divided by pool_size.')
        return T.max(x.reshape(x.shape[0], x.shape[1]/self.pool_size,
                               self.pool_size), axis=2)


def maxout(x, pool_size=4):
    if x.shape[1] % pool_size == 0:
        raise ValueError('x.shape[1] must be divided by pool_size.')
    return T.max(x.reshape(x.shape[0], x.shape[1]/pool_size, pool_size),
                 axis=2)


class ELU(Activation):
    def __init__(self, layer=None, alpha=1.0):
        super(ELU, self).__init__(layer)
        self.alpha = alpha

    def __call__(self, x):
        return T.switch(x > 0, x, self.alpha*(T.exp(x)-1.))


def elu(x, alpha=1.0):
    return T.switch(x > 0, x, alpha*(T.exp(x)-1.))


def get_activation(identifier):
    return utils.get_from_module(identifier, globals(), 'activations')
