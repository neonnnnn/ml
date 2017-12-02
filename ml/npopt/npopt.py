from ..utils import get_from_module
from abc import ABCMeta, abstractmethod


class Optimizer(object):
    __metaclass__ = ABCMeta
    def __init__(self, clipping=None):
        if isinstance(clipping, (int, float)):
            self.clipping = (-clipping, clipping)
        elif isinstance(clipping, (tuple, list)):
            self.clipping = clipping
        else:
            self.clipping = None

    @abstractmethod
    def get_updates(self, grad, param):
        pass

    def clip_gradient(self, grad):
        if self.clipping is not None:
            grad = np.clip(grad, *self.clipping)
        return grad


class SGD(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.9, clipping=None):
        self.lr = lr
        self.momentum = momentum
        self.ms = {}
        super(SGD, self).__init__(clipping)

    def get_update(self, grad, param):
        grad = self.clip_gradient(grad)
        update = -self.lr*grad
        return update


class MomentumSGD(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.9, clipping=None):
        self.lr = lr
        self.momentum = momentum
        self.ms = {}
        for param in params:
            self.ms.update({id(param):np.zeros(param.shape)})
        super(SGD, self).__init__(clipping)

    def get_update(self, grad, param):
        grad = self.clip_gradient(grad)
        update = -self.lr*grad + self.momentum * self.ms[id(param)]
        self.ms[id(param)] = updates
        return update


class AdaGrad(Optimizer):
    def __init__(self, params, lr=0.01, eps=1e-8, clipping=None):
        self.lr = lr
        self.eps = eps
        self.accumulate_gradient = {}
        for param in params:
            self.accumulate_gradient.update({id(param):np.zeros(param.shape)+0})
        super(AdaGrad, self).__init__(clipping)

    def get_update(self, grad, param):
        grad = self.clip_gradient(grad)
        self.accumulate_gradient[id(param)] += grad**2
        update = -self.lr * grad / (np.sqrt(self.accumulate_gradient[id(param)])+self.eps)
        return update

def get_optimizer(identifier, lr, params):
    return get_from_module(identifier, globals(), 'optimizer', True, kwargs={'params':params, 'lr':lr})


sgd = SGD
momentumsgd = MomentumSGD
adagrad = Adagrad = AdaGrad
