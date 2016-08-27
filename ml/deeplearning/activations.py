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
    return T.nnet.relu(x, alpha)


def get_activation(identifier):
    return utils.get_from_module(identifier, globals(), 'activations')


relu = ReLU
leakyrelu = LeakyReLU
