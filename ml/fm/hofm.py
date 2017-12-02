import numpy as np
from .anova import anova, grad_anova
from .fm import FactorizationMachine, sigmoid
from ..npopt.npopt import get_optimizer
from sklearn.metrics import accuracy_score

class HOFM(FactorizationMachine):
    def __init__(self,
                 order=3,
                 k=30,
                 reg=[1e-5,1e-5],
                 task='r',
                 optimizer='adagrad',
                 lr=0.01,
                 sigma=0.1,
                 seed=1,
                 iprint=True):
        self.order = order
        super(HOFM, self).__init__(k, reg, task, optimizer, lr, sigma, seed, iprint)
        self.P = None

    def init_params(self, d):
        self.P = [self.rng.normal(0, self.sigma, (d, self.k)).T for _ in range(self.order-1)]
        self.w = self.rng.normal(0, self.sigma, (d,))
        self.b  = np.zeros(1)

        if self.optimizer != 'cd' and self.optimizer != 'als':
            self._optimizer = self._stoc_update
            if isinstance(self.optimizer, Optimizer):
                self.optimizer = get_optimizer(self.optimizer.__name__, 
                                               lr=self.lr, 
                                               params=self.P+[self.w, self.b])
            else:
                self.optimizer = get_optimizer(self.optimizer, lr=self.lr, params=self.P+[self.w, self.b])
        else:
            self._optimizer = self._coordinate_descent

    def _anova(self, X):
        if X.ndim == 1:
            d = X.shape[0]
            n = 1
        else:
            n, d = X.shape
        anova_table = np.zeros((self.order+1, d+1))
        pred = np.zeros((n))
        for i, x in enumerate(np.atleast_2d(X)):
            for m, p in enumerate(self.P):
                for pk in p:
                    pred[i] += anova(pk, x, m+2, anova_table)
        return pred

    def _stoc_grad(self, x, y, m, dptable_anova, dptable_grad, sum_anova_this):
        grad_loss_P = np.zeros((self.k, x.shape[0]))
        cur_anova = 0
        for s in range(self.k):
            cur_anova += anova(self.P[m-2][s], x, m, dptable_anova)
            grad_loss_P[s] = grad_anova(self.P[m-2][s], x, m, dptable_anova, dptable_grad)

        y_pred = self.b + np.dot(x, self.w) + sum_anova_this + cur_anova
        grad_loss_f = self.grad_loss_f(y, y_pred)
        grad_loss_w = grad_loss_f * x + self.reg[0] * self.w
        grad_loss_P *= grad_loss_f
        grad_loss_P += self.reg[1] * self.P[m-2]

        return [grad_loss_f, grad_loss_w, grad_loss_P]

    def _stoc_update(self, X_train, y_train, m, dptable_anova, dptable_grad, sum_anova):
        # sum_anova: summantion of anova kernel except order m.
        # sum_anova is used for efficient computation of y_pred
        for idx in self.rng.permutation(X_train.shape[0]):
            x = X_train[idx]
            y = y_train[idx]
            grads  = self._stoc_grad(x, y, m, dptable_anova, dptable_grad, sum_anova[idx])
            self.b += self.optimizer.get_update(grads[0], self.b)
            self.w += self.optimizer.get_update(grads[1], self.w)
            self.P[m-2] += self.optimizer.get_update(grads[2], self.P[m-2])
            # update sum_anova[i]
            for p in self.P[m-2]:
                sum_anova[idx] += anova(p, x, m, dptable_anova)

    def _coordinate_descent(self, X_train, y_train, m, dptable_anova, dptable_grad, sum_anova):
        raise NotImplementedError('TODO')

    def fit(self, X_train, y_train, n_epoch=10000):
        d = X_train.shape[1]
        if self.P is None:
            self.init_params(d)

        if self.task == 'c' and (np.max(y_train) != 1 or np.min(y_train) != -1):
            raise ValueError('When task is "c", y_train must be {1, -1}^n.')

        print('training...')
        self._fit(X_train, y_train, n_epoch)

        print('training complete')

    def _fit(self, X_train, y_train, n_epoch):
        d = X_train.shape[1]
        if self.optimizer != 'CD':
            opt_method = self._stoc_update
            dptable_anova = np.zeros((self.order+1, d+1))
            dptable_grad = np.zeros((self.order+1, d))

            sum_anova = self._anova(X_train)
            for epoch in range(n_epoch):
                # When a stochastic gradient based method is used, only P[m-2]
                # (i.e., the weights for m-order feature conjunctions) is optimized in each epoch.
                m = (epoch % (self.order-1)) + 2
                for i, x in enumerate(X_train):
                    for p in self.P[m-2]:
                        sum_anova[i] -= anova(p, x, m, dptable_anova)
                opt_method(X_train, y_train, m, dptable_anova, dptable_grad, sum_anova)
                if self.iprint:
                    output = self.b + np.dot(X_train, self.w) + sum_anova
                    loss = np.mean(self.loss(output, y_train))
                    if self.task == 'c':
                        acc = accuracy_score(y_train, np.sign(pred))
                        print('  epoch:{0} loss:{1} accuracy:{2}'.format(epoch, loss, acc))
                    else:
                        print('  epoch:{0} loss:{1}'.format(epoch, loss))
        else:
            # TODO: Implmenting the CD method.
            opt_method = self._coordinate_descent

