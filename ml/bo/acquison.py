from __future__ import division
from .. import utils
import numpy as np
import scipy


class MI(object):
    def __init__(self, delta=1e-3):
        self.sqrt_alpha = np.sqrt(np.log(2. / delta))
        self.gamma_t = 0

    def calc(self, mean, var):
        phi = self.sqrt_alpha*(np.sqrt(var+self.gamma_t)
                               - np.sqrt(self.gamma_t))
        next_idx = np.argmax(mean + phi)
        self.gamma_t += var
        self.gamma_t = np.delete(self.gamma_t, next_idx, 0)

        return next_idx


class UCB(object):
    def __init__(self, nu=1, delta=1e-3):
        self.nu = nu
        self.delta = delta
        self.t = 0
        self.d = None
        
    def calc(self, mean, var):
        self.t += 1
        tau = 2 * ((self.d/2.+2)*np.log(self.t)
                   + 2*np.log(np.pi) - np.log(3*self.delta))
        next_idx = np.argmax(mean + np.sqrt(self.nu*tau*var))

        return next_idx


class PI(object):
    def __init__(self, xi=0):
        self.xi = xi
        self.best = None

    def calc(self, mean, var):
        phi = scipy.stats.norm.cdf((mean - self.best - self.xi) / np.sqrt(var))
        phi[np.where(var == 0)] = 0
        next_idx = np.argmax(phi)

        return next_idx


class EI(object):
    def __init__(self):
        self.best = None

    def calc(self, mean, var):
        gamma = (mean - self.best) / np.sqrt(var)
        ei = np.sqrt(var) * (gamma*scipy.stats.norm.cdf(gamma)
                             + scipy.stats.norm.pdf(gamma))
        ei[np.where(var == 0)] = 0
        next_idx = np.argmax(ei)

        return next_idx


def get_acquison(identifier):
    return utils.get_from_module(identifier, globals(), 'acquison')
