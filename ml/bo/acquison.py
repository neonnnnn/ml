from __future__ import division
from .. import utils
import numpy as np


class MI(object):
    def __init__(self, delta=1e-3):
        self.sqrt_alpha = np.sqrt(np.log(2. / delta))
        self.gamma = 0

    def calc(self, mean, var):
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma) - np.sqrt(self.gamma))
        next_idx = np.argmax(mean + phi)
        self.gamma += var

        return next_idx


class UCB(object):
    def __init__(self, nu=1, delta=1e-3):
        self.nu = nu
        self.delta = delta
        self.t = 0
        self.d = None
        
    def calc(self, mean, var):
        tau = 2 * ((self.d / 2 + 2) * np.log(self.t) + 2*np.log(np.pi) - np.log(3 * self.delta))
        next_idx = np.argmax(mean + np.sqrt(self.nu * tau) * var)

        return next_idx


def get_acquison(identifier):
    return utils.get_from_module(identifier, globals(), 'acquison')
