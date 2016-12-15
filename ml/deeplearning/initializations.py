from __future__ import absolute_import
import numpy as np
from .. import utils


def uniform(layer, shape, scale=0.05):
    return layer.rng.uniform(low=-scale, high=scale, size=shape)


def normal(layer, shape, mu=0., sigma=0.02):
    return layer.rng.normal(mu, sigma, shape)


def zero(layer, shape):
    return np.zeros(shape)


def glorot_uniform(layer, shape, coeff=6.):
    if len(shape) == 2:
        scale = np.sqrt(coeff / sum(shape))
    else:
        fan_in = np.prod(layer.filter_shape[1:])
        fan_out = layer.filter_shape[0] * np.prod(layer.filter_shape[2:])
        scale = np.sqrt(coeff / (fan_in + fan_out))
    return uniform(layer, shape, scale)


def lecun_uniform(layer, shape):
    scale = np.sqrt(3. / layer.fan_in)
    return uniform(layer, shape, scale)


def he_conv_normal(layer, shape):
    mu = 0
    sigma = np.sqrt(2. / (layer.n_in[0] * (layer.filter_shape[0] ** 2)))
    return normal(layer, shape, mu, sigma)


def get_init(identifier):
    return utils.get_from_module(identifier, globals(), 'initializations')
