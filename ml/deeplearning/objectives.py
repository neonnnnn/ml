import theano.tensor as T
from abc import ABCMeta, abstractmethod


class Loss(object):
    __metaclass__ = ABCMeta

    def __init__(self, weight=1, mode='mean'):
        self.weight = weight
        if mode == 'sum':
            self.mode = 0
        elif mode == 'mean':
            self.mode = 1
        elif mode == 0 or mode == 1:
            self.mode = mode
        else:
            raise ValueError('mode must be 0 or "sum", or 1 or "mean".')

    def __call__(self, *args):
        self.calc(*args)

    @abstractmethod
    def calc(self, **kwargs):
        pass


class Regularization(object):
    __metaclass__ = ABCMeta

    def __init__(self, weight=1e-5):
        self.weight = weight

    def __call__(self, *args):
        self.calc(*args)

    @abstractmethod
    def calc(self, **kwargs):
        pass


class CrossEntropy(Loss):
    def calc(self, y, output):
        if y.ndim == 1:
            loss = -(y * T.log(T.clip(output.ravel(), 1e-20, 1))
                     + (1 - y) * T.log(T.clip(1 - output.ravel(), 1e-20, 1)))
        else:
            axis = tuple(range(y.ndim))[1:]
            loss = -T.sum((y * T.log(T.clip(output, 1e-20, 1))
                           + (1 - y) * T.log(T.clip(1 - output, 1e-20, 1))),
                          axis=axis)

        if self.mode:
            loss = T.mean(loss)
        else:
            loss = T.sum(loss)
        return self.weight * loss


class MulticlassLogLoss(Loss):
    def calc(self, y, output):
        # if categorical variables
        if y.ndim == 1:
            loss = -(T.log(1e-20 + output)[T.arange(y.shape[0]), y])
        # if one-hot
        else:
            axis = tuple(range(y.ndim))[1:]
            loss = -(T.sum(y * T.log(1e-20 + output), axis=axis))
        if self.mode:
            loss = T.mean(loss)
        else:
            loss = T.sum(loss)
        return self.weight * loss


class Hinge(Loss):
    def calc(self, y, output):
        if y.ndim == 1:
            loss = T.maximum(1. - y * output.ravel(), 0.)
        else:
            axis = tuple(range(y.ndim))[1:]
            loss = T.sum(T.maximum(1. - y * output.ravel(), 0.), axis=axis)
        if self.mode:
            loss = T.mean(loss)
        else:
            loss = T.sum(loss)
        return self.weight * loss


class KLDivergence(Loss):
    def calc(self, y, output):
        loss = T.sum(y * T.log(T.clip(y / output, 1e-20, 1)), axis=1)
        if self.mode:
            loss = T.mean(loss)
        else:
            loss = T.sum(loss)
        return self.weight * loss


class SquaredError(Loss):
    def calc(self, y, output):
        if y.ndim == 1:
            loss = (T.square(y - output))
        else:
            axis = tuple(range(y.ndim))[1:]
            loss = T.sum(T.square(y - output), axis=axis)
        if self.mode:
            loss = T.mean(loss)
        else:
            loss = T.sum(loss)
        return self.weight * loss


class L2Regularization(Regularization):
    def calc(self, layers):
        reg = 0
        for layer in layers:
            if hasattr(layer, 'W'):
                reg += T.sum(layer.W ** 2)
        return self.weight * reg


class L1Regularization(Regularization):
    def calc(self, layers):
        reg = 0
        for layer in layers:
            if hasattr(layer, 'W'):
                reg += T.sum(abs(layer.W))
        return self.weight * reg


class KLDivergenceRegularization(Regularization):
    def __init__(self, idx, rho=0.2, weight=1e-5):
        super(KLDivergenceRegularization, self).__init__(weight)
        self.idx = idx
        if type(idx, int) != 'int':
            raise TypeError('the type of idx must be int.')
        self.rho = rho

    def calc(self, layers):
        reg = T.sum(self.rho * T.log(T.clip(self.rho / layers[self.idx].W,
                                            1e-20, 1)))
        return self.weight * reg
