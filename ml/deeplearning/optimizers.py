import numpy as np
import theano
import theano.tensor as T


class SGD(object):
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = theano.shared(np.asarray(lr, dtype=theano.config.floatX))
        self.momentum = theano.shared(np.asarray(momentum, dtype=theano.config.floatX))

    def get_update(self, cost, params):
        grads = T.grad(cost, params)
        updates = []

        for p, g in zip(params, grads):
            m = theano.shared(value=np.asarray(np.zeros(p.get_value().shape), dtype=theano.config.floatX))
            next_m = -self.lr * g + self.momentum * m
            updates.append((m, next_m))
            next_p = p + next_m
            updates.append((p, next_p))

        return updates


class AdaGrad(object):
    def __init__(self, lr=1., eps=1.):
        self.lr = theano.shared(np.asarray(lr, dtype=theano.config.floatX))
        self.eps = eps

    def get_update(self, cost, params):
        grads = T.grad(cost, params)
        updates = []
        accumulate_gradient = [theano.shared(value=np.asarray(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]

        for p, g, a_g in zip(params, grads, accumulate_gradient):
            next_a_g = a_g + T.sqr(g)
            updates.append((a_g, next_a_g))

            next_p = p - self.lr * g / T.sqrt(a_g + self.eps)
            updates.append((p, next_p))

        return updates


class AdaDelta(object):
    def __init__(self, lr=1., eps=1e-6, rho=0.95):
        self.lr = theano.shared(np.asarray(lr, dtype=theano.config.floatX))
        self.eps = theano.shared(np.asarray(eps, dtype=theano.config.floatX))
        self.rho = theano.shared(np.asarray(rho, dtype=theano.config.floatX))

    def get_update(self, cost, params):
        grads = T.grad(cost, params)
        updates = []
        accumulate_gradient = [theano.shared(value=np.asarray(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]
        accumulate_updates = [theano.shared(value=np.asarray(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]

        for p, g, a_g, a_u in zip(params, grads, accumulate_gradient, accumulate_updates):
            next_a_g = self.rho * a_g + (1 - self.rho) * T.sqr(g)
            updates.append((a_g, next_a_g))

            delta_params = g * T.sqrt(a_u + self.eps) / T.sqrt(next_a_g + self.eps)
            next_p = p - self.lr * delta_params
            updates.append((p, next_p))

            next_a_u = self.rho * a_u + (1 - self.rho) * T.sqr(delta_params)
            updates.append((a_u, next_a_u))

        return updates


class RMSprop(object):
    def __init__(self, lr=0.001, rho=0.9, eps=1e-6):
        self.lr = theano.shared(np.asarray(lr, dtype=theano.config.floatX))
        self.rho = theano.shared(np.asarray(rho, dtype=theano.config.floatX))
        self.eps = eps

    def get_update(self, cost, params):
        grads = T.grad(cost, params)
        accumulate_gradient = [theano.shared(value=np.asarray(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]
        updates = []

        for p, g, a_g in zip(params, grads, accumulate_gradient):
            new_a_g = self.rho * a_g + (1 - self.rho) * T.square(g)
            updates.append((a_g, new_a_g))

            new_p = p - self.lr * g / T.sqrt(new_a_g + self.eps)
            updates.append((p, new_p))

        return updates


class Adam(object):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
        self.lr = theano.shared(np.asarray(lr, dtype=theano.config.floatX))
        self.beta_1 = theano.shared(np.asarray(beta_1, dtype=theano.config.floatX))
        self.beta_2 = theano.shared(np.asarray(beta_2, dtype=theano.config.floatX))
        self.eps = theano.shared(np.asarray(eps, dtype=theano.config.floatX))

    def get_update(self, cost, params):
        grads = T.grad(cost, params)
        updates = []

        i = theano.shared(value=np.asarray(0, dtype=theano.config.floatX))
        updates.append((i, i+1))

        t = i+1
        lr_t = self.lr * T.sqrt(1 - T.pow(self.beta_2, t)) / (1 - T.pow(self.beta_1, t))
        eps_hat = self.eps * T.sqrt(1 - self.beta_2 ** t)
        ms = [theano.shared(value=np.asarray(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]
        vs = [theano.shared(value=np.asarray(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * T.sqr(g)
            p_t = p - lr_t * m_t / (T.sqrt(v_t) + eps_hat)

            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))

        return updates


class AdaMax(object):
    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999):
        self.lr = theano.shared(np.asarray(lr, dtype=theano.config.floatX))
        self.beta_1 = theano.shared(np.asarray(beta_1, dtype=theano.config.floatX))
        self.beta_2 = theano.shared(np.asarray(beta_2, dtype=theano.config.floatX))

    def get_update(self, cost, params):
        grads = T.grad(cost, params)
        updates = []

        i = theano.shared(value=np.asarray(0, dtype=theano.config.floatX))
        updates.append((i, i + 1))
        
        t = i+1
        lr_t = self.lr / (1. - T.pow(self.beta_1, t))
        ms = [theano.shared(value=np.asarray(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]
        us = [theano.shared(value=np.asarray(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]

        for p, g, m, u in zip(params, grads, ms, us):
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            u_t = T.maximum(self.beta_2 * u, T.abs_(g))
            p_t = p - lr_t * m_t / (u_t + 1e-8)

            updates.append((m, m_t))
            updates.append((u, u_t))
            updates.append((p, p_t))

        return updates
