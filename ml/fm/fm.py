import numpy as np
import sys
from ..npopt.npopt import get_optimizer


def sigmoid(x):
    return np.exp(np.minimum(0, x)) / (1 + np.exp(-abs(x)))


class FactorizationMachine(object):
    def __init__(self, k,
                 reg=[1e-5, 1e-5],
                 task='r',
                 optimizer='adagrad',
                 lr=0.01,
                 sigma=0.01,
                 seed=1,
                 iprint=True):
        self.k = k
        self.reg = reg
        self.lr = lr
        self.b = None
        self.w = None
        self.V = None
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)
        self.optimizer = optimizer
        self._optimizer = None
        self.task = task
        self.iprint = iprint
        if task == 'r' or task == 'regression':
            self.task = 'r'
            self.loss = self._squared_loss
            self.grad_loss_f = self._grad_squared_f
            self.mu = 1
        elif task == 'c' or task == 'classification':
            self.task = 'c'
            self.loss = self._logit_loss
            self.grad_loss_f = self._grad_logit_f
            self.mu = 1./4.
        else:
            raise ValueError('task is must be "r" or "regression", or "c" or "classification".') 

   def init_params(self, d):
        self.V = self.rng.normal(0, self.sigma, (d, self.k))
        self.w = self.rng.normal(0, self.sigma, (d,))
        self.b = np.zeros(1)
        if self.optimizer != 'cd' and self.optimizer != 'als':
            self._optimizer = self._stoc_update
            if isinstance(self.optimizer, Optimizer):
                self.optimizer = get_optimizer(self.optimizer.__name__, 
                                               lr=self.lr, 
                                               params=[self.V, self.w, self.b])
            else:
                self.optimizer = get_optimizer(self.optimizer, lr=self.lr, params= [self.V, self.w, self.b])
        else:
            self._optimizer = self._coordinate_descent

    def predict(self, x):
        output = self.decision_function(x)
        if self.task == 'c':
            output = np.sign(output)
        return output

    def _anova(self, x):
        return 0.5*(np.sum(np.atleast_2d(np.dot(x, self.V)**2 - np.dot(x**2, self.V**2)), axis=1))

    def decision_function(self, x):
        return self.b + np.dot(x, self.w) + self._anova(x)

    @staticmethod
    def _logit_loss(output, y):
        return np.log(np.exp(-abs(y*output))+1) - np.minimum(0, y*output)

    @staticmethod
    def _grad_logit_f(y, output):
        return y*(sigmoid(y*output) - 1)

    @staticmethod
    def _squared_loss(output, y):
        return 0.5*(output-y) ** 2

    @staticmethod
    def _grad_squared_f(y, output):
        return (output-y)

    def _grad_f_V(self, xi):
        return np.outer(xi, (np.dot(xi, self.V))) - self.V*np.atleast_2d(xi**2).T

    def _stoc_grad(self, x, y, y_pred):
        grad_loss_f = self.grad_loss_f(y, y_pred)
        grad_loss_w = grad_loss_f*x + self.reg[0]*self.w
        grad_loss_V = grad_loss_f*self._grad_f_V(x) + self.reg[1]*self.V        
        return [grad_loss_f, grad_loss_w, grad_loss_V]

    def _stoc_update(self, X_train, y_train):
        for xi, yi in zip(X_train, y_train):
            output = self.decision_function(xi)
            grads = self._stoc_grad(xi, yi, output)
            self.b += self.optimizer.get_update(grads[0], self.b)
            self.w += self.optimizer.get_update(grads[1], self.w)
            self.V += self.optimizer.get_update(grads[2], self.V)

    def _coordinate_descent(self, X_train, y_train):
        # this is the coordinate descent(CD) algorihtm proposed by Mathieu Blondel et al.
        # Paper titile: Polynomial Networks and Factorization Machines: New Insights and Efficient Training Algorithms
        # When task = "r", the CD algorithms is equivalent to the alternative least squares(ALS) algotihm implemented in libFM.
        n, d = X_train.shape
        idxs = [np.where(X_train[:, i] > 0) for i in range(d)]
        output = self.decision_function(X_train)
        grad_loss_f = self.grad_loss_f(output, y_train)
        # update b
        b_new = self.b  - np.sum(grad_loss_f) / (self.mu * n)
        output += b_new - self.b
        self.b = b_new
        grad_loss_f = self.grad_loss_f(output, y_train)
        # update w
        grad_f_w = X_train.T
        grad_f_w_squared_sum = np.sum(grad_f_w**2, axis=1)
        for i in range(d):
            wi_new = self.w[i]*self.mu*grad_f_w_squared_sum[i] - np.dot(grad_f_w[i], grad_loss_f)
            wi_new /=  self.mu*grad_f_w_squared_sum[i]+self.reg[0]
            output += (wi_new - self.w[i])*X_train[:, i]
            self.w[i] = wi_new
            grad_loss_f = self.grad_loss_f(output, y_train)
        # update V
        for f in range(self.k):
            q_f = np.dot(X_train, self.V[:, f])
            # i means l in original paper: "Factorization Machines with libFM"
            for i,(idx, xi) in enumerate(zip(idxs, X_train.T)):
                conjunc_other = q_f[idx] - xi[idx]*self.V[i,f]
                grad_f_v = xi[idx] * conjunc_other
                grad_f_v_squared_sum = np.sum(grad_f_v**2)
                vif_new = self.V[i,f]*self.mu*grad_f_v_squared_sum - np.dot(grad_f_v, grad_loss_f[idx])
                vif_new /= self.mu*grad_f_v_squared_sum + self.reg[1]
                output[idx] += (vif_new - self.V[i, f])*xi[idx]*conjunc_other
                q_f[idx] += (vif_new-self.V[i, f]) * xi[idx]
                self.V[i, f] = vif_new
                grad_loss_f = self.grad_loss_f(output, y_train)

    def fit(self, X_train, y_train, n_epoch):
        if self.V is None:
            self.init_params(X_train.shape[1])
        if self.task == 'c' and (np.max(y_train) != 1 or np.min(y_train) != -1):
            raise ValueError('When task is "c", y_train must be {1, -1}^n.')
        print('training...')
        self._fit(X_train, y_train, n_epoch)

        print('training complete')

    def _fit(self, X_train, y_train, n_epoch):
        for i in range(n_epoch):
            idx = self.rng.permutation(y_train.shape[0])
            X_train, y_train = X_train[idx], y_train[idx]
            self._optimizer(X_train, y_train)
            if self.iprint:
                output = self.decision_function(X_train)
                loss = np.mean(self.loss(output, y_train))
                if self.task == 'c':
                    pred = np.sign(output)
                    acc = accuracy_score(y_train, pred)
                    print('  epoch:{0} loss:{1} accuracy:{2}'.format(i, loss, acc))
                else:
                    print('  epoch:{0} loss:{1}'.format(i, loss))
