from __future__ import absolute_import
import numpy as np
import theano
import theano.tensor as T
from ml.utils import progbar
import timeit
import scipy.sparse as sp
from theano.compile.sharedvalue import SharedVariable


def sharedzeros(shape, name=None):
    return theano.shared(np.zeros(shape, dtype=theano.config.floatX), borrow=True, name=name)


def sharedones(shape, name=None):
    return theano.shared(np.ones(shape, dtype=theano.config.floatX), borrow=True, name=name)


def sharedasarray(input, name=None):
    if input is None or isinstance(input, SharedVariable):
        return input
    else:
        return theano.shared(np.asarray(input, dtype=theano.config.floatX), borrow=True, name=name)


def variable(x, name=None):
    if x.ndim == 1:
        if x.dtype == theano.config.floatX:
            ret = T.vector(name=name)
        elif np.issubdtype(x.dtype, np.integer):
            ret = T.ivector(name=name)
    elif x.ndim == 2:
        ret = T.matrix(name=name)
    else:
        ret = getattr(T, 'tensor'+str(x.ndim))(name=name)
    return ret


def log_sum_exp(x, axis=0, keepdims=False):
    x_max = T.max(x, axis=axis, keepdims=True)
    _x_max = T.max(x, axis=axis, keepdims=keepdims)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=keepdims)) + _x_max


def log_mean_exp(x, axis=0, keepdims=False):
    x_max = T.max(x, axis=axis, keepdims=True)
    _x_max = T.max(x, axis=axis, keepdims=keepdims)
    return T.log(T.mean(T.exp(x - x_max), axis=axis, keepdims=keepdims)) + _x_max


# running 1 epoch
def run(inputs, function, iprint=True):
    output = []
    if iprint:
        n_batches = inputs.n_batches
        s = timeit.default_timer()
        i = 0
    for batch in inputs:
        output += function(*[b if not sp.issparse(b) else b.toarray() for b in batch])
        if iprint:
            e = timeit.default_timer()
            progbar(i+1, n_batches, e-s)
            i += 1
    return output


def run_on_batch(inputs, function):
    output = function(*[b if not sp.issparse(b) else b.toarray() for b in inputs])
    return output


