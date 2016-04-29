from __future__ import division
import utils
import numpy as np
from scipy import sparse

# return shape : (x2.shape[0], x1.shape[1])


class Linear(object):
    def __init__(self, params=None):
        pass

    @staticmethod
    def calc_kernel(x1, x2):
        return x2.dot(x1.T)

    @staticmethod
    def calc_kernel_diagonal(x1):
        return x1.dot(x1)


class Polynomial(object):
    def __init__(self, params):
        self.d = params[0]
        self.c = params[1]

    def calc_kernel(self, x1, x2):
        return (x2.dot(x1.T) + self.c) ** self.d

    def calc_kernel_diagonal(self, x1):
        return (x1.dot(x1) + self.c) ** self.d


class RBF(object):
    def __init__(self, params):
        self.gamma = params[0]

    def calc_kernel(self, x1, x2):
        x2 = np.atleast_2d(x2)
        norm1 = np.ones([x2.shape[0], 1]) * np.sum(x1 ** 2, axis=1)
        norm2 = np.ones([x1.shape[0], 1]) * np.sum(x2 ** 2, axis=1)
        return np.exp(- self.gamma * (norm1 + norm2.T - 2 * x2.dot(x1.T)))

    @staticmethod
    def calc_kernel_diagonal(x1):
        return 1.0


class Pairwise(object):
    def __init__(self, params):
        self.vec_len = params[0]

    def calc_kernel(self, x1, x2):
        k1 = (x2[:self.vec_len]).dot(x1[:self.vec_len].T) * (x2[self.vec_len:]).dot(x1[self.vec_len:].T)
        k2 = (x2[:self.vec_len]).dot(x1[self.vev_len:].T) * (x2[self.vec_lem:]).dot(x1[:self.vec_len].T)

        return (k1 + k2) / 2.

    def calc_kernel_diag(self, x1):
        k1 = (x1[:self.vec_len]).dot(x1[:self.vec_len]) * (x1[self.vec_len:]).dot(x1[self.vec_len:])
        k2 = (x1[:self.vec_len]).dot(x1[self.vec_len:]) * (x1[self.vec_len:]).dot(x1[:self.vec_len])
        return (k1 + k2) / 2.


def get_kernel(identifier):
    return utils.get_from_module(identifier, globals(), 'kernel')

linear = Linear
polynomial = Polynomial
rbf = gaussian = Gaussian = gauss = Gauss = RBF
pairwise = PairWise = Pairwise