import theano.tensor as T
from .. import utils


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def softmax(x):
    return T.nnet.softmax(x)


def relu(x):
    return T.nnet.relu(x)


def leakyrelu(x, alpha=0.2):
    if alpha <= 0 or alpha >= 1:
        raise ValueError('0 < alpha < 1.')
    if isinstance(alpha, tuple):
        return T.nnet.relu(x, alpha[0])
    else:
        return T.nnet.relu(x, alpha)


def maxout(x, pool_size=4):
    if pool_size <= 0:
        raise ValueError('pool_size must be Natural number.')
    if x.shape[1] % pool_size == 0:
        raise ValueError('x.shape[1] must be divided by pool_size.')

    return T.max(x.reshape(x.shape[0], x.shape[1]/pool_size, pool_size), axis=2)


def elu(x, alpha=1.0):
    if isinstance(alpha, tuple):
        return T.switch(x > 0, x, alpha[0] * (T.exp(x) - 1.))
    else:
        return T.switch(x > 0, x, alpha * (T.exp(x) - 1.))


def get_activation(identifier):
    return utils.get_from_module(identifier, globals(), 'activations')


ReLU = relu
LeakyReLU = LReLU = leakyrelu
