from __future__ import division
from .. import utils
import numpy as np


class Linear(object):
    def __init__(self, params=None):
        pass

    @staticmethod
    def calc(x1, x2):
        return np.dot(x1, x2.T).T

    @staticmethod
    def calc_kernel_same(x1):
        return np.sum(x1*x1, axis=1)


class SparseLinear(object):
    def __init__(self, params=None):
        pass

    @staticmethod
    def calc(x1, x2):
        return x1.dot(x2.T).T.toarray()

    @staticmethod
    def calc_same(x1):
        return np.array(x1.multiply(x1).sum(axis=1)).ravel()


class Polynomial(object):
    def __init__(self, params):
        self.d = params[0]
        self.c = params[1]

    def calc(self, x1, x2):
        return (np.dot(x1, x2.T).T+self.c) ** self.d

    def calc_same(self, x1):
        return (np.sum(x1*x1, axis=1)+self.c) ** self.d


class SparsePolynomial(object):
    def __init__(self, params):
        self.d = params[0]
        self.c = params[1]

    def calc(self, x1, x2):
        return (x1.dot(x2.T).T.toarray()+self.c) ** self.d

    def calc_same(self, x1):
        return (np.array(x1.multiply(x1).sum(axis=1)).ravel()+self.c) ** self.d


class RBF(object):
    def __init__(self, params):
        self.gamma = params[0]

    def calc(self, x1, x2):
        x2 = np.atleast_2d(x2)
        return np.exp(-self.gamma * (np.atleast_2d(np.sum(x1*x1, axis=1))
                                     + np.atleast_2d(np.sum(x2*x2, axis=1)).T
                                     - 2 * np.dot(x1, x2.T).T))

    @staticmethod
    def calc_same(x1):
        return np.ones(x1.shape[0])


class SparseRBF(object):
    def __init__(self, params):
        self.gamma = params[0]

    def calc_kernel(self, x1, x2):
        norm1 = np.array(x1.multiply(x1).sum(axis=1)).ravel()
        norm2 = np.array(x2.multiply(x2).sum(axis=1)).ravel()
        return np.exp(- self.gamma * (np.ones([x2.shape[0], 1])*norm1
                                      + (np.ones([x1.shape[0], 1])*norm2).T
                                      - 2*x1.dot(x2.T).T.toarray()))

    @staticmethod
    def calc_kernel_same(x1):
        return np.ones(x1.shape[0])


def get_kernel(identifier):
    return utils.get_from_module(identifier, globals(), 'kernel')

linear = Linear
Sparselinear = SparseLinear
polynomial = Polynomial
Sparsepolynomial = SparsePolynomial
rbf = RBF
Sparserbf = SparseRBF

