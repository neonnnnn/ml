import numpy as np
import sys
from scipy import sparse


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


class FactorizationMachine(object):
    def __init__(self, k, reg=[1e-5, 1e-5, 1e-5], task='r', optimizer='sgd',
                 iter=10000, lr=0.001, sigma=0.1, seed=1):
        self.k = k
        self.reg = reg
        self.iter = iter
        self.lr = lr
        self.bias = 0
        self.w = None
        self.V = None
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)
        self.optimizer = optimizer
        self.decision_function = None
        self.task = task
        if task == 'r' or task == 'regression':
            self.loss = self._square_loss
            self.grad = self._grad_square_loss
        elif task == 'c' or task == 'classification':
            self.loss = self._sigmoid_crossentropy_loss
            self.grad = self._grad_sigmoid_crossentropy_loss
        else:
            raise ValueError('task is must be "r" or "regression", '
                             'or "c" or "classification".')

    def predict(self, x):
        if self.task == 'r':
            return self.decision_function(self, x)
        else:
            return np.sign(self.decision_function(x))

    def decision_function(self, x):
        return (0.5*(np.sum(np.atleast_2d(x.dot(self.V)**2
                            - (x**2).dot(self.V**2)), axis=1))
                + x.dot(self.w) + self.bias)

    def _decision_function_sparse(self, x, nnz):
        return (0.5*(np.sum((np.dot(x, self.V[nnz, :])**2
                             - np.dot(x**2, self.V[nnz, :]**2))))
                + np.dot(x, self.w[nnz]) + self.bias)

    @staticmethod
    def _sigmoid_crossentropy_loss(output, y):
        return -(y*np.log(sigmoid(output)) + (1-y)*np.log(1.-sigmoid(output)))

    @staticmethod
    def _grad_sigmoid_crossentropy_loss(output, y):
        return sigmoid(output) - y

    @staticmethod
    def _square_loss(output, y):
        return (output-y) ** 2

    @staticmethod
    def _grad_square_loss(output, y):
        return 2 * (output-y)

    def _grad_V_output(self, xi):
        return (np.dot(np.atleast_2d(xi).T, np.atleast_2d(np.dot(xi, self.V)))
                - self.V*np.atleast_2d(xi**2).T)

    def _grad_V_output_sprse(self, xi, nnz):
        return (np.dot(np.atleast_2d(xi).T,
                       np.atleast_2d(np.dot(xi, self.V[nnz])))
                - self.V[nnz]*np.atleast_2d(xi**2).T)

    def _sgd(self, X_train, y_train):
        for xi, yi in zip(X_train, y_train):
            output = self.decision_function(xi)
            grad_loss_output = self.grad(output, yi)
            self.bias -= self.lr * grad_loss_output
            self.w -= self.lr * (grad_loss_output*xi+self.reg[1]*self.w)
            self.V -= self.lr * (grad_loss_output*self._grad_V_output(xi)
                                 + self.reg[2]*self.V)

    def _sgd_sparse(self, find, y_train, indexs):
        for i in indexs:
            nnz = find[1][find[0] == i]
            xi = find[2][find[0] == i]
            output = self._decision_function_sparse(xi, nnz)
            grad_loss_output = self.grad(output, y_train[i])
            self.bias -= self.lr * grad_loss_output
            self.w[nnz] -= self.lr * grad_loss_output*xi
            self.V[nnz] -= (self.lr * grad_loss_output
                            * self._grad_V_output_sprse(xi, nnz))
            self.w -= self.lr * self.reg[1]*self.w
            self.V -= self.lr * self.reg[2]*self.V

    def _adagrad(self, X_train, y_train):
        a_g_b = 0
        a_g_w = np.zeros(self.w.shape)
        a_g_V = np.zeros(self.V.shape)
        eps = 1e-6
        for xi, yi in zip(X_train, y_train):
            output = self.decision_function(xi)
            grad_loss_output = self.grad(output, yi)
            grad_b = grad_loss_output
            grad_w = grad_loss_output*xi + self.reg[1]*self.w
            grad_V = (grad_loss_output*self._grad_V_output(xi)
                      + self.reg[2]*self.V)
            a_g_b += grad_b ** 2
            a_g_w += grad_w ** 2
            a_g_V += grad_V ** 2
            self.b -= self.lr * grad_b / np.sqrt(a_g_b+eps)
            self.w -= self.lr * grad_w / np.sqrt(a_g_w+eps)
            self.V -= self.lr * grad_V / np.sqrt(a_g_V+eps)

    def _adagrad_sparse(self, X_train, y_train):
        a_g_b = 0
        a_g_w = np.zeros(self.w.shape)
        a_g_V = np.zeros(self.V.shape)
        eps = 1e-6
        for xi, yi in zip(X_train, y_train):
            output = self.decision_function(xi)
            grad_loss_output = self.grad(output, y_train)
            grad_b = grad_loss_output
            grad_w = grad_loss_output*xi + self.reg[1]*self.w
            grad_V = (grad_loss_output*self._grad_V_output(xi)
                      + self.reg[2]*self.V)
            a_g_b += grad_b ** 2
            a_g_w += grad_w ** 2
            a_g_V += grad_V ** 2
            self.bias -= self.lr * grad_b / np.sqrt(a_g_b+eps)
            self.w -= self.lr * grad_w / np.sqrt(a_g_w+eps)
            self.V -= self.lr * grad_V / np.sqrt(a_g_V+eps)

    def _als(self, X_train, y_train):
        n = y_train.shape[0]
        output = self.decision_function(X_train)
        e = y_train - output
        new_bias = (self.bias*n+np.sum(e)) / (n+self.reg[0])
        e += new_bias - self.bias
        self.bias = new_bias

        x_squared = X_train**2
        x_squared_sum_axis0 = np.sum(x_squared, axis=0)
        for i in xrange(n):
            new_wi = ((self.w[i]*x_squared_sum_axis0[i]
                       + np.dot(X_train[:, i], e))
                      / (x_squared_sum_axis0[i]+self.reg[1]))
            e += (new_wi-self.w[i]) * X_train[:, i]
            self.w[i] = new_wi

        for i in xrange(self.k):
            q = np.dot(X_train, self.V[:, i])
            for j in xrange(n):
                grad_vji = X_train * (q-X_train[:, j]*self.V[j, i])
                new_v = ((self.V[j, i]*np.sum(grad_vji**2)
                          + np.dot(grad_vji, e))
                         / (np.sum(grad_vji**2)+self.reg[2]))
                q += (new_v-self.V[j, i]) * X_train[:, j]
                e += (new_v-self.V[j, i]) * grad_vji
                self.V[j, i] = new_v

    def fit(self, X_train, y_train):
        self.V = self.rng.normal(0, self.sigma, (X_train.shape[1], self.k))
        self.w = np.zeros(X_train.shape[1])
        self.bias = 0.0

        print('training...')
        if sparse.issparse(X_train):
            self._fit_sparse(X_train, y_train, sparse.find(X_train))
        else:
            self._fit(X_train, y_train)

        print('training complete')

    def _fit(self, X_train, y_train):
        for i in xrange(self.iter):
            idx = self.rng.permutation(y_train.shape[0])
            X_train, y_train = X_train[idx], y_train[idx]
            self.optimizer(X_train, y_train)

    def _fit_sparse(self, X_train, y_train, find):
        for i in xrange(self.iter):
            idx = self.rng.permutation(y_train.shape[0])
            self.optimizer(find, y_train, idx)

    SGD = sgd = _sgd
    ALS = als = _als
    AdaGrad = adagrad = _adagrad
    sgd_sparse = _sgd_sparse
    adagrad_sparse = _adagrad_sparse
