import theano
import numpy as np


def sharedzeros(shape):
    return theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                         borrow=True)


def sharedones(shape):
    return theano.shared(np.ones(shape, dtype=theano.config.floatX),
                         borrow=True)


def sharedasarray(input):
    return theano.shared(np.asarray(input, dtype=theano.config.floatX),
                         borrow=True)
