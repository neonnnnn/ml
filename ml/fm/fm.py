import numpy as np


class FactorizationMachine(object):
    def __init__(self, k, reg=[1e-5, 1e-5, 1e-5], task='r', optimizer='sgd',
                 iter=10000, lr=0.0001, sigma=0.01,
                 rng=np.random.RandomState(1)):
        self.k = k
        self.reg = reg
        self.iter = iter
        self.lr = lr
        self.bias = 0
        self.w = None
        self.V = None
        self.sigma = sigma
        self.rng = rng
        self.optimizer = self.__getattribute__(optimizer)
        if task == 'r' or task == 'regression':
            self.loss = self._sigmoid_crossentropy_loss
            self.grad = self._grad_sigmoid_crossentropy_loss
        elif task == 'c' or task == 'classification':
            self.loss = self._square_loss
            self.grad = self._grad_square_loss
        else:
            raise ValueError('task is must be "r" or "regression", '
                             'or "c" or "classification".')

    def predict(self, x):
        if self.task == 'r':
            return self.decision_function(self, x)
        else:
            return np.sign(self.decision_function(x))

    def decision_function(self, x):
        return (0.5*(np.sum(np.atleast_2d(np.dot(x, self.V)**2), axis=1)
                     - np.sum(np.atleast_2d(np.dot(x**2, self.V**2)), axis=1))
                + np.dot(x, self.w) + self.bias)

    @staticmethod
    def _sigmoid_crossentropy_loss(output, y):
        return -(y * np.log(1. / (1 + np.exp(-output)))
                 + (1 - y) * np.log(1. - 1 / (1 + np.exp(-output))))

    @staticmethod
    def _grad_sigmoid_crossentropy_loss(output, y):
        return 1. / (1 + np.exp(-output)) - y

    @staticmethod
    def _square_loss(output, y):
        return (output - y) ** 2

    @staticmethod
    def _grad_square_loss(output, y):
        return 2 * (output - y)

    def _grad_V_output(self, xi):
        return (np.dot(np.atleast_2d(xi).T, np.atleast_2d(np.dot(xi, self.V)))
                - self.V * np.atleast_2d(xi ** 2).T)

    def _sgd(self, x, y):
        for xi, yi in zip(x, y):
            output = self.decision_function(x)
            grad_loss_output = self.grad(output, y)
            self.bias -= self.lr * (grad_loss_output + self.reg[0] * self.bias)
            self.bias -= self.lr * (grad_loss_output * xi + self.reg[1] * self.w)
            self.V -= self.lr * (grad_loss_output * self._grad_V_output(xi)
                                 + self.reg[2] * self.V)

    def _als(self, x, y):
        n = y.shape[0]
        output = self.decision_function(x)
        e = y - output
        new_bias = (self.bias * n + np.sum(e)) / (n + self.reg[0])
        e += new_bias - self.bias
        self.bias = new_bias

        x_squared = x**2
        x_squared_sum_axis0 = np.sum(x_squared, axis=0)
        for i in xrange(n):
            new_wi = ((self.w[i]*x_squared_sum_axis0[i]+np.dot(x[:, i], e))
                      / (x_squared_sum_axis0[i] + self.reg[1]))
            e += (new_wi - self.w[i]) * x[:, i]
            self.w[i] = new_wi

        for i in xrange(self.k):
            q = np.dot(x, self.V[:, i])
            for j in xrange(n):
                grad_vji = x * (q - x[:, j] * self.V[j, i])
                new_v = ((self.V[j, i] * np.sum(grad_vji ** 2)
                          + np.dot(grad_vji, e))
                         / (np.sum(grad_vji ** 2) + self.reg[2]))
                q += (new_v - self.V[j, i]) * x[:, j]
                e += (new_v - self.V[j, i]) * grad_vji
                self.V[j, i] = new_v

    def fit(self, x, y):
        self.V = self.rng.normal(0, self.sigma, (x.shape[1], self.k))
        self.w = np.zeros(x.shape[0])
        self.bias = 0.0

        print 'training...'
        for i in xrange(self.iter):
            idx = self.rng.permutation(y.shape[0])
            x, y = x[idx], y[idx]
            self.optimizer(x, y)

        print 'training complete'

    SGD = sgd = _sgd
    ALS = als = _als
