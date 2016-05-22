from __future__ import division
import utils
import numpy as np
# return shape : (x2.shape[0], x1.shape[0])
# return type: numpy.ndarray


def calc_r2(x1, x2, theta):
    x1 = x1 * np.sqrt(theta)
    x2 = x2 * np.sqrt(theta)
    norm1 = np.sum(x1 ** 2, axis=1)
    norm2 = np.sum(x2 ** 2, axis=1)[:, np.newaxis]
    r2 = -2 * np.dot(x2, x1.T) + norm1 + norm2
    return np.where(r2 < 0, 0, r2)


class Matern52(object):
    def __init__(self, theta=None):
        self.theta = theta
        self.dim = 0
        self.ard = True

    def calc_kernel(self, x1, x2, theta=None):
        if theta is None:
            theta = self.theta
        r2 = calc_r2(x1, x2, theta)

        r = np.sqrt(r2)
        return (1 + np.sqrt(5) * r + 5. * r2 / 3) * np.exp(-np.sqrt(5) * r)

    @staticmethod
    def calc_kernel_same(x1):
        return 1.0

    @staticmethod
    def calc_kernel_diag(x1):
        return np.ones(x1.shape[0])

    def calc_grad(self, x1, x2, theta):
        n = x1.shape[0]
        grad_r2 = np.zeros((x1.shape[1], n, n))
        for i in range(self.dim):
            grad_r2[i] = theta[i] * (x1[:, i] - (x1[:, i])[:, np.newaxis]) ** 2
        grad_r2 += 1e-10

        r2 = calc_r2(x1, x2, theta)
        r = np.sqrt(r2)
        inv_r = 1. / np.where(r != 0, r, np.inf)
        sqrt5 = np.sqrt(5)
        grad1 = (sqrt5 * grad_r2 * inv_r / 2. + 5. * grad_r2 / 3.)
        grad2 = (-sqrt5 * grad_r2 * inv_r / 2.) * (1 + sqrt5 * r + 5. * r2 / 3.)

        return (grad1 + grad2) * np.exp(-sqrt5 * r2)



matern52=Matern52
