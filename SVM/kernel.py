from __future__ import division
import utils
import numpy as np
# return shape : (x2.shape[0], x1.shape[1])
# return type: numpy.ndarray


class Linear(object):
    def __init__(self, params=None):
        pass

    @staticmethod
    def calc_kernel(x1, x2):
        return np.dot(x2, x1.T)

    @staticmethod
    def calc_kernel_same(x1):
        return np.dot(x1, x1)


class SparseLinear(object):
    def __init__(self, params=None):
        pass

    @staticmethod
    def calc_kernel(x1, x2):
        return x1.dot(x2.T).T.toarray()

    @staticmethod
    def calc_kernel_same(x1):
        return x1.multiply(x1).sum()


class Polynomial(object):
    def __init__(self, params):
        self.d = params[0]
        self.c = params[1]

    def calc_kernel(self, x1, x2):
        return (np.dot(x2, x1.T) + self.c) ** self.d

    def calc_kernel_same(self, x1):
        return (np.dot(x1, x1) + self.c) ** self.d


class SparsePolynomial(object):
    def __init__(self, params):
        self.d = params[0]
        self.c = params[1]

    def calc_kernel(self, x1, x2):
        return (x1.dot(x2.T).T.toarray() + self.c) ** self.d

    def calc_kernel_same(self, x1):
        return (x1.multiply(x1).sum() + self.c) ** self.d


class RBF(object):
    def __init__(self, params):
        self.gamma = params[0]

    def calc_kernel(self, x1, x2):
        x2 = np.atleast_2d(x2)
        norm1 = np.ones([x2.shape[0], 1]) * np.sum(x1 ** 2, axis=1)
        norm2 = np.ones([x1.shape[0], 1]) * np.sum(x2 ** 2, axis=1)
        return np.exp(- self.gamma * (norm1 + norm2.T - 2 * np.dot(x2, x1.T)))

    @staticmethod
    def calc_kernel_same(x1):
        return 1.0


class SparseRBF(object):
    def __init__(self, params):
        self.gamma = params[0]

    def calc_kernel(self, x1, x2):
        norm1 = np.array(x1.multiply(x1).sum(axis=1)).ravel()
        norm2 = np.array(x2.multiply(x2).sum(axis=1)).ravel()
        return np.exp(- self.gamma * (np.ones([x2.shape[0], 1]) * norm1 + (np.ones([x1.shape[0], 1]) * norm2).T - 2 * x1.dot(x2.T).T.toarray()))

    @staticmethod
    def calc_kernel_same(x1):
        return 1.0


def get_kernel(identifier):
    return utils.get_from_module(identifier, globals(), 'kernel')

linear = Linear
Sparselinear = SparseLinear
polynomial = Polynomial
Sparsepolynomial = SparsePolynomial
rbf = RBF
Sparserbf = SparseRBF

