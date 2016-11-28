from __future__ import division
from .. import utils
import numpy as np
# return shape : (x2.shape[0], x1.shape[0])
# return type: numpy.ndarray


def calc_r2(x1, x2, theta):
    x1 = x1 / theta
    x2 = x2 / theta
    norm1 = np.sum(x1 ** 2, axis=1)
    norm2 = np.sum(x2 ** 2, axis=1)[:, np.newaxis]
    r2 = -2 * np.dot(x2, x1.T) + norm1 + norm2
    return np.where(r2 < 0, 0, r2)


def calc_grad_r2(x1, x2, theta):
    n = x1.shape[0]
    grad_r2 = np.zeros((theta.shape[0], n, n))
    for i in range(0, theta.shape[0]):
        grad_r2[i] = -2 * (x1[:, i - 1] - (x2[:, i - 1])[:, np.newaxis]) ** 2
        grad_r2[i] /= theta[i] ** 3

    return grad_r2


class SE(object):
    def __init__(self, theta=None):
        self.theta = theta
        self.dim = 1
        self.ard = True

    def calc_kernel(self, x1, x2, theta=None):
        if theta is None:
            theta = self.theta
        r2 = calc_r2(x1, x2, theta[1:])
        return theta[0] * np.exp(-0.5 * r2)

    def calc_grad(self, x1, x2, theta):
        n = x1.shape[0]
        grad = np.zeros((self.dim, n, n))
        grad_r2 = calc_grad_r2(x1, x1, theta[1:])

        r2 = calc_r2(x1, x2, theta[1:])
        grad[1:] = -0.5 * grad_r2 * theta[0]
        grad[0] = 1
        return grad * np.exp(-0.5 * r2)

    def calc_kernel_diag(self, x1):
        return self.theta[0] * np.ones(x1.shape[0])


class Matern52(object):
    def __init__(self, theta=None):
        self.theta = theta
        self.dim = 1
        self.ard = True

    def calc_kernel(self, x1, x2, theta=None):
        if theta is None:
            theta = self.theta
        r2 = calc_r2(x1, x2, theta[1:])

        r = np.sqrt(r2)
        return theta[0] * (1+np.sqrt(5)*r+5.*r2/3.) * np.exp(-np.sqrt(5)*r)

    def calc_kernel_diag(self, x1):
        return np.ones(x1.shape[0]) * self.theta[0]

    def calc_grad(self, x1, x2, theta):
        r2 = calc_r2(x1, x2, theta[1:])
        r = np.sqrt(r2)
        n = x1.shape[0]
        grad = np.zeros((self.dim, n, n))
        grad_r2 = calc_grad_r2(x1, x2, theta[1:])

        sqrt5 = np.sqrt(5)
        grad[1:] = theta[0] * grad_r2 * -5./6. * (1+sqrt5*r)
        grad[0] = 1 + np.sqrt(5)*r + 5.*r2/3.

        return grad * np.exp(-sqrt5 * r)


def get_kernel(identifier):
    return utils.get_from_module(identifier, globals(), 'kernel')

matern52 = Matern52
