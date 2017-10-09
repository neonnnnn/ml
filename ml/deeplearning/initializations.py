from __future__ import absolute_import
from __future__ import absolute_import
import numpy as np
from ..utils import get_from_module


def uniform(scale=0.05):
    def _uniform(layer, shape):
        return layer.rng.uniform(low=-scale, high=scale, size=shape)
    return _uniform


def normal(mu=0., sigma=0.001):
    def _normal(layer, shape):
        return layer.rng.normal(mu, sigma, shape)
    return _normal


def zero():
    def _zero(layer, shape):
        return np.zeros(shape)
    return _zero


def glorot_uniform(coeff=6.):
    def _glorot_uniform(layer, shape):
        if len(shape) == 2:
            scale = np.sqrt(coeff / sum(shape))
        else:
            fan_in = np.prod(layer.filter_shape[1:])
            fan_out = layer.filter_shape[0] * np.prod(layer.filter_shape[2:])
            scale = np.sqrt(coeff / (fan_in + fan_out))
        return uniform(scale)(layer, shape)
    return _glorot_uniform


def lecun_uniform():
    def _lecun_uniform(layer, shape):
        scale = np.sqrt(3. / layer.fan_in)
        return uniform(scale)(layer, shape)
    return _lecun_uniform


def he_conv_normal():
    def _he_conv_normal(layer, shape):
        mu = 0
        sigma = np.sqrt(2. / (layer.n_in[0] * (layer.filter_shape[0] ** 2)))
        return normal(mu, sigma)(layer, shape)
    return _he_conv_normal


def get_init(identifier):
    return get_from_module(identifier, globals(), 'initializations')
