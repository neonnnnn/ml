from __future__ import absolute_import
import numpy as np
from .. import utils


def uniform(rng, shape, scale=0.05):
    return rng.uniform(low=-scale, high=scale, size=shape)


def normal(rng, shape, mu=0., sigma=0.05):
    return rng.normal(mu, sigma, shape)


def glorot_uniform(layers, shape):
    scale = np.sqrt(6. / (layers.fan_in + layers.fan_out))
    return uniform(layers.rng, shape, scale)


def lecun_uniform(layers, shape):
    scale = np.sqrt(3. / layers.fan_in)
    return uniform(layers.rng, shape, scale)


def he_conv_normal(layers, shape):
    mu = 0
    sigma = np.sqrt(2. / (layers.n_in[0] * (layers.filter_shape[0] ** 2)))
    return normal(layers.rng, shape, mu, sigma)


def get_init(identifier):
    return utils.get_from_module(identifier, globals(), 'initializations')
