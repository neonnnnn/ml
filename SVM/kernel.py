from __future__ import division
import utils
import numpy as np


# return shape : (x2.shape[0], x1.shape[0])

def linear(x1, x2, params):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    return np.dot(x2, x1.T)


def polynomial(x1, x2, params):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    return (np.dot(x2, x1.T) + params[0]) ** (params[1])


def rbf(x1, x2, params):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    norm1 = np.ones([x2.shape[0], 1]) * np.sum(x1 ** 2, axis=1)
    norm2 = np.ones([x1.shape[0], 1]) * np.sum(x2 ** 2, axis=1)
    return np.exp(- params[0] * (norm1 + norm2.T - 2 * np.dot(x2, x1.T)))


def get_kernel(identifier):
    return utils.get_from_module(identifier, globals(), 'kernel')

Linear = linear
Polynomial = polynomial
RBF = gaussian = Gaussian = gauss = Gauss = rbf


if __name__=='__main__':
    a = np.array([[1,2,3], [4,5,6]])
    b = np.array([7,8,9])
    c = polynomial(a, b, [1, 4])
    print c