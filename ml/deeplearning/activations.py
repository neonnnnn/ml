import theano.tensor as T
from .. import utils


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def softmax(x):
    return T.nnet.softmax(x)


def ReLU(x):
    return T.nnet.relu(x)


def LeakyReLU(x, alpha=0.2):
    if isinstance(alpha, tuple):
        return T.nnet.relu(x, alpha[0])
    else:
        return T.nnet.relu(x, alpha)


def elu(x, alpha=1.0):
    if isinstance(alpha, tuple):
        return T.switch(x > 0, x, alpha[0] * (T.exp(x) - 1.))
    else:
        return T.switch(x > 0, x, alpha * (T.exp(x) - 1.))


def get_activation(identifier):
    return utils.get_from_module(identifier, globals(), 'activations')


relu = ReLU
leakyrelu = LeakyReLU
