from __future__ import absolute_import
import numpy as np
from .. import utils


def uniform(layers, shape, scale=0.05):
    return layers.rng.uniform(low=-scale, high=scale, size=shape)


def normal(layers, shape, mu=0., sigma=0.02):
    return layers.rng.normal(mu, sigma, shape)


def glorot_uniform(layers, shape):
    scale = np.sqrt(6. / (layers.fan_in + layers.fan_out))
    return uniform(layers, shape, scale)


def lecun_uniform(layers, shape):
    scale = np.sqrt(3. / layers.fan_in)
    return uniform(layers, shape, scale)


def he_conv_normal(layers, shape):
    mu = 0
    sigma = np.sqrt(2. / (layers.n_in[0] * (layers.filter_shape[0] ** 2)))
    return normal(layers, shape, mu, sigma)


def get_init(identifier):
    return utils.get_from_module(identifier, globals(), 'initializations')
