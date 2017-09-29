from __future__ import absolute_import
import theano.tensor as T
from .theanoutils import sharedasarray, sharedzeros
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
    def get_updates(self, cost, params):
        pass

    def get_gradients(self, cost, params):
        grads = T.grad(cost, params)
        if self.clipping is not None:
            grads = [T.clip(g, *self.clipping) for g in grads]
        return grads


class SGD(Optimizer):
    def __init__(self, lr=0.001, momentum=0.9, clipping=None):
        self.lr = sharedasarray(lr)
        self.momentum = sharedasarray(momentum)
        self.ms = None
        super(SGD, self).__init__(clipping)

    def get_updates(self, cost, params):
        grads = self.get_gradients(cost, params)
        updates = []
        if self.ms is None:
            self.ms = [sharedzeros(p.get_value().shape) for p in params]

        for p, g, m in zip(params, grads, self.ms):
            next_m = -self.lr*g + self.momentum*m
            updates.append((m, next_m))
            next_p = p + next_m
            updates.append((p, next_p))

        return updates


class AdaGrad(Optimizer):
    def __init__(self, lr=0.01, eps=1e-6, clipping=None):
        self.lr = sharedasarray(lr)
        self.eps = sharedasarray(eps)
        self.accumulate_gradient = None
        super(AdaGrad, self).__init__(clipping)

    def get_updates(self, cost, params):
        grads = self.get_gradients(cost, params)
        updates = []
        if self.accumulate_gradient is None:
            self.accumulate_gradient = [sharedzeros(p.get_value().shape) for p in params]

        for p, g, a_g in zip(params, grads, self.accumulate_gradient):
            next_a_g = a_g + T.sqr(g)
            updates.append((a_g, next_a_g))

            next_p = p - self.lr*g/(T.sqrt(a_g+self.eps)+self.eps)
            updates.append((p, next_p))

        return updates


class AdaDelta(Optimizer):
    def __init__(self, lr=0.01, eps=1e-6, rho=0.95, clipping=None):
        self.lr = sharedasarray(lr)
        self.eps = sharedasarray(eps)
        self.rho = sharedasarray(rho)
        self.accumulate_gradient = None
        self.accumulate_updates = None
        super(AdaDelta, self).__init__(clipping)

    def get_updates(self, cost, params):
        grads = self.get_gradients(cost, params)
        updates = []
        if self.accumulate_gradient is None:
            self.accumulate_gradient = [sharedzeros(p.get_value().shape) 
                                        for p in params]
        if self.accumulate_updates is None:
            self.accumulate_updates = [sharedzeros(p.get_value().shape)
                                       for p in params]

        for p, g, a_g, a_u in zip(params, grads, self.accumulate_gradient,
                                  self.accumulate_updates):
            next_a_g = self.rho*a_g + (1-self.rho)*T.sqr(g)
            updates.append((a_g, next_a_g))

            delta_params = g*T.sqrt(a_u+self.eps) / T.sqrt(next_a_g+self.eps)
            next_p = p - self.lr*delta_params
            updates.append((p, next_p))

            next_a_u = self.rho*a_u + (1-self.rho)*T.sqr(delta_params)
            updates.append((a_u, next_a_u))

        return updates


class RMSprop(Optimizer):
    def __init__(self, lr=0.001, rho=0.9, eps=1e-6, clipping=None):
        self.lr = sharedasarray(lr)
        self.rho = sharedasarray(rho)
        self.eps = eps
        self.accumulate_gradient = None
        super(RMSprop, self).__init__(clipping)
        
    def get_updates(self, cost, params):
        grads = self.get_gradients(cost, params)
        if self.accumulate_gradient is None:
            self.accumulate_gradient = [sharedzeros(p.get_value().shape)
                                        for p in params]
        updates = []

        for p, g, a_g in zip(params, grads, self.accumulate_gradient):
            new_a_g = self.rho*a_g + (1-self.rho)*T.square(g)
            updates.append((a_g, new_a_g))

            new_p = p - self.lr*g/T.sqrt(new_a_g+self.eps)
            updates.append((p, new_p))

        return updates


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, clipping=None):
        self.lr = sharedasarray(lr)
        self.beta1 = sharedasarray(beta1)
        self.beta2 = sharedasarray(beta2)
        self.eps = sharedasarray(eps)
        self.i = None
        self.ms = None
        self.vs = None
        super(Adam, self).__init__(clipping)

    def get_updates(self, cost, params):
        grads = self.get_gradients(cost, params)
        updates = []
        if self.i is None:
            self.i = sharedasarray(0)
        updates.append((self.i, self.i+1))

        t = self.i+1
        lr_t = self.lr * T.sqrt(1-self.beta2**t) / (1-self.beta1**t)
        eps_hat = self.eps * T.sqrt(1-self.beta2**t)
        if self.ms is None:
            self.ms = [sharedzeros(p.get_value().shape) for p in params]
        if self.vs is None:
            self.vs = [sharedzeros(p.get_value().shape) for p in params]

        for p, g, m, v in zip(params, grads, self.ms, self.vs):
            m_t = (self.beta1*m) + (1.-self.beta1)*g
            v_t = (self.beta2*v) + (1.-self.beta2)*(g**2)
            p_t = p - lr_t*m_t/(T.sqrt(v_t)+eps_hat)

            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))

        return updates

    def get_updates_value(self, cost, params):
        grads = self.get_gradients(cost, params)
        updates = []
        if self.i is None:
            self.i = sharedasarray(0)

        t = self.i+1
        lr_t = self.lr * T.sqrt(1-self.beta2**t) / (1-self.beta1**t)
        eps_hat = self.eps * T.sqrt(1-self.beta2**t)
        if self.ms is None:
            self.ms = [sharedzeros(p.get_value().shape) for p in params]
        if self.vs is None:
            self.vs = [sharedzeros(p.get_value().shape) for p in params]

        for p, g, m, v in zip(params, grads, self.ms, self.vs):
            if self.clipping is not None:
                g = T.clip(g, *self.clipping)
            m_t = (self.beta1*m) + (1.-self.beta1)*g
            v_t = (self.beta2*v) + (1.-self.beta2)*(g**2)
            updates.append(lr_t*m_t/(T.sqrt(v_t)+eps_hat))

        return updates


class AdaMax(Optimizer):
    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, clipping=None):
        self.lr = sharedasarray(lr)
        self.beta1 = sharedasarray(beta1)
        self.beta2 = sharedasarray(beta2)
        self.i = None
        self.ms = None
        self.us = None
        super(AdaMax, self).__init__(clipping)

    def get_updates(self, cost, params):
        grads = self.get_gradients(cost, params)
        updates = []
        if self.i is None:
            self.i = sharedasarray(0)
            updates.append((self.i, self.i+1))
        
        t = self.i+1
        lr_t = self.lr / (1.-T.pow(self.beta1, t))
        if self.ms is None:
            self.ms = [sharedzeros(p.get_value().shape) for p in params]
        if self.us is None:
            self.us = [sharedzeros(p.get_value().shape) for p in params]

        for p, g, m, u in zip(params, grads, self.ms, self.us):
            m_t = self.beta1*m + (1.-self.beta1)*g
            u_t = T.maximum(self.beta2*u, T.abs_(g))
            p_t = p - lr_t*m_t/(u_t+1e-8)

            updates.append((m, m_t))
            updates.append((u, u_t))
            updates.append((p, p_t))

        return updates
