import numpy as np
import theano
import theano.tensor as T


class MulticlassLogLoss(object):
    def __init__(self, weight=1.):
        self.weight = weight

    def get_output(self, y, p_y_given_x):
        # if categorical variables
        if y.ndim == 1:
            loss = -T.mean(T.log(1e-20 + p_y_given_x)[T.arange(y.shape[0]), y])
        # if one-hot
        elif y.ndim == 2:
            loss = -T.mean(T.sum(y * T.log(1e-20 + p_y_given_x), axis=1))
        # else
        else:
            raise Exception('Label Error:label must be scalar or vector. If not miss, you must rewrite model, objective etc.')

        return self.weight * loss

    def get_output_no_symbol(self, y, p_y_given_x):
        # if categorical variables
        if y.ndim == 1:
            loss = -np.mean(np.log(1e-20 + p_y_given_x)[np.arange(y.shape[0]), y])
        # if one-hot
        elif y.ndim == 2:
            loss = -np.mean(np.sum(y * np.log(1e-20 + p_y_given_x), axis=1))
        # else
        else:
            raise Exception('Label Error:label must be scalar or vector. If not miss, you must rewrite model, objective etc.')
        return self.weight * loss


class KL(object):
    def __init__(self, weight=1.):
        self.weight = weight

    def get_output(self, p, q):
        loss = T.mean(p * T.log(p / q) + (1 - p) * T.log((1 - p) / (1 - q)))
        return self.weight * loss

    def get_output_no_symbol(self, p, q):
        loss = np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))
        return self.weight * loss


class MeanSquaredError(object):
    def __init__(self, weight=1.):
        self.weight = weight

    def get_output(self, y, p_y_given_x):
        loss = T.mean(T.sum(T.square(y - p_y_given_x), axis=1))
        return self.weight * loss

    def get_output_no_symbol(self, y, p_y_given_x):
        loss = np.mean(np.sum(np.square(y - p_y_given_x), axis=1))
        return self.weight * loss


class L2Regularization(object):
    def __init__(self, weight=0.0001):
        self.weight = weight

    def get_output(self, layers):
        n_layers = len(layers)
        L2_reg = 0
        for i in xrange(n_layers):
            if hasattr(layers[i], 'W'):
                L2_reg += (layers[i].W ** 2).sum()

        return self.weight * L2_reg


class L1Regularization(object):
    def __init__(self, weight=0.0001):
        self.weight = weight

    def get_output(self, layers):
        n_layers = len(layers)
        L1_reg = 0
        for i in xrange(n_layers):
            if hasattr(layers[i], 'W'):
                L1_reg += abs(layers[i].W).sum()

        return self.weight * L1_reg



