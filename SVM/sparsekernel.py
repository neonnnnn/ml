from __future__ import division
import utils
import numpy as np
# calc_kernel: return scipy.sparse.csc_matrix, (x2.shape[0], x1.shape[1])
# calc_kernel_same: return numpy.float64


class Linear(object):
    def __init__(self, params=None):
        pass

    @staticmethod
    def calc_kernel(x1, x2):
        return x1.dot(x2.T).T

    @staticmethod
    def calc_kernel_same(x1):
        return np.array(x1.multiply(x1).sum(axis=1)).ravel()


class Polynomial(object):
    def __init__(self, params):
        if len(params) > 1:
            print ("When using sparsekernel.Polynomial, coef is ignored.")
        self.d = params[0]

    def calc_kernel(self, x1, x2):
        return x1.dot(x2.T).T.power(self.d)

    def calc_kernel_same(self, x1):
        return np.array(x1.multiply(x1).sum(axis=1)).ravel() ** self.d


class Certesian(object):
    def __init__(self, params):
        self.vec_len = params[0]

    def calc_kernel(self, x1, x2):
        k1 = (x1[:, :self.vec_len].dot(x2[:, :self.vec_len].T)).multiply((x1[:, self.vec_len:]).dot(x2[:, self.vec_len:].T))
        k2 = (x1[:, self.vec_len:].dot(x2[:, :self.vec_len].T)).multiply((x1[:, :self.vec_len]).dot(x2[:, self.vec_len:].T))
        return (k1 + k2).T / 2.

    def calc_kernel_same(self, x1):
        k1 = np.multiply(np.array(x1[:, :self.vec_len].multiply(x1[:, :self.vec_len]).sum(axis=1)).ravel(),
                         np.array(x1[:, self.vec_len:].multiply(x1[:, self.vec_len:]).sum(axis=1)).ravel())
        k2 = np.multiply(np.array(x1[:, :self.vec_len].multiply(x1[:, self.vec_len:]).sum(axis=1)).ravel(),
                         np.array(x1[:, self.vec_len:].multiply(x1[:, :self.vec_len]).sum(axis=1)).ravel())
        return (k1 + k2) / 2.


def get_kernel(identifier):
    return utils.get_from_module(identifier, globals(), 'sparsekernel')


linear = Linear
polynomial = Polynomial
certesian = Certesian
