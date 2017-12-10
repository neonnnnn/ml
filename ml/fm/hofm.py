import numpy as np
from .anova import anova, grad_anova, anova_alt, grad_anova_alt, anova_saving_memory
from .fm import FactorizationMachine
from ..npopt.npopt import get_optimizer, Optimizer


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

        if not self.optimizer in ['cd', 'CD', 'als', 'ALS']:
            self._optimizer = self._stoc_update
            if isinstance(self.optimizer, Optimizer):
                self.optimizer = get_optimizer(self.optimizer.__name__, 
                                               lr=self.lr, 
                                               params=self.P+[self.w, self.b])
            else:
                self.optimizer = get_optimizer(self.optimizer,
                                               lr=self.lr,
                                               params=self.P+[self.w, self.b])
        else:
            self._optimizer = self._coordinate_descent

    def _anova(self, X):
        if X.ndim == 1:
            d = X.shape[0]
            n = 1
        else:
            n, d = X.shape
        anova_vector = np.zeros(self.order+1)
        pred = np.zeros(n)
        for i, x in enumerate(np.atleast_2d(X)):
            for m, p in enumerate(self.P):
                for pk in p:
                    pred[i] += anova_saving_memory(pk, x, m+2, anova_vector)
        return pred

    def _stoc_grad(self, x, y, m, dptable_anova, dptable_grad, sum_anova_this):
        grad_loss_P = np.zeros((self.k, x.shape[0]))
        cur_anova = 0
        for s in range(self.k):
            cur_anova += anova(self.P[m-2][s], x, m, dptable_anova)
            grad_loss_P[s] = grad_anova(self.P[m-2][s], x, m, dptable_anova, dptable_grad)

        output = self.b + np.dot(x, self.w) + sum_anova_this + cur_anova
        grad_loss_f = self.grad_loss_f(y, output)
        grad_loss_w = grad_loss_f * x + self.reg[0] * self.w
        grad_loss_P *= grad_loss_f
        grad_loss_P += self.reg[1]*self.P[m-2]

        return [grad_loss_f, grad_loss_w, grad_loss_P]

    def _stoc_update(self, X_train, y_train, m, dptable_anova, dptable_grad, sum_anova):
        # sum_anova: summantion of anova kernel except order m.
        # sum_anova is used for efficient computation of y_pred
        for i, x in enumerate(X_train):
            for p in self.P[m-2]:
                sum_anova[i] -= anova(p, x, m, dptable_anova)
        for idx in self.rng.permutation(X_train.shape[0]):
            x = X_train[idx]
            y = y_train[idx]
            grads  = self._stoc_grad(x, y, m, dptable_anova, dptable_grad, sum_anova[idx])
            self.b += self.optimizer.get_update(grads[0], self.b)
            self.w += self.optimizer.get_update(grads[1], self.w)
            self.P[m-2] += self.optimizer.get_update(grads[2], self.P[m-2])
        for i, x in enumerate(X_train):
            for p in self.P[m-2]:
                sum_anova[i] += anova(p, x, m, dptable_anova)

        return None

    def _coordinate_descent(self,
                            X_train,
                            y_train,
                            m,
                            dptable_anova,
                            dptable_grad,
                            dptable_poly,
                            output,
                            idxs):
        # this is the coordinate descent(CD) algorihtm proposed by Mathieu Blondel et al.
        # When task = "r", the CD algorithms is equivalent to the ALS implemented in libFM.
        # Preliminary: computing output and grad_loss_f
        n, d = X_train.shape
        grad_loss_f = self.grad_loss_f(y_train, output)

        # update b
        b_new = self.b - np.sum(grad_loss_f) / (self.mu * n)
        output += b_new - self.b
        self.b = b_new
        grad_loss_f = self.grad_loss_f(y_train, output)

        # update w
        grad_f_w = X_train.T
        grad_f_w_squared_sum = np.sum(grad_f_w**2, axis=1)
        for i, idx in enumerate(idxs):
            wi_new = self.w[i]*self.mu*grad_f_w_squared_sum[i]
            wi_new -= np.dot(grad_f_w[i, idx], grad_loss_f[idx])
            wi_new /= self.mu*grad_f_w_squared_sum[i]+self.reg[0]
            output[idx] += (wi_new - self.w[i])*X_train[idx, i]
            self.w[i] = wi_new
            grad_loss_f[idx] = self.grad_loss_f(y_train[idx], output[idx])
        # update P^{m}
        for s in range(self.k):
            # pre-computing D^{t}(p_js, x_i) \forall i \in [n], t \ in [m]
            for t in range(1, m+1):
                dptable_poly[:, t] = np.dot(X_train**t, self.P[m-2][s]**t)
            # update p_{js} for m-order anova kernel = self.P[m-2][s, j]
            for j, (idx, xj) in enumerate(zip(idxs, X_train.T)):
                if len(idx) == 0:
                    continue
                # computing m-order anova kernel and its gradient wrt p_{js}
                anova_alt(self.P[m-2][s], X_train[idx], m, dptable_anova[idx], dptable_poly[idx])
                grad_anova_pjs = grad_anova_alt(self.P[m-2][s, j],
                                                xj[idx],
                                                m,
                                                dptable_anova[idx],
                                                dptable_poly[idx],
                                                dptable_grad[idx])
                eta = self.mu * np.sum(grad_anova_pjs**2) + n*self.reg[1]
                new_pjs = (self.P[m-2][s, j]
                           - (np.dot(grad_loss_f[idx], grad_anova_pjs)+n*self.P[m-2][s, j])/eta)
                # update D~{t} t \in [m] and output
                for t in range(1, m+1):
                    dptable_poly[idx, t] += (new_pjs**t - self.P[m-2][s, j]**t) * xj[idx]**t
                output[idx] -= dptable_anova[idx, m]
                self.P[m-2][s,j] = new_pjs
                anova_alt(self.P[m-2][s,j], xj, m, dptable_anova[idx], dptable_poly[idx])
                output[idx] += dptable_anova[idx, m]
                grad_loss_f[idx] = self.grad_loss_f(y_train[idx], output[idx])

        return output

    def fit(self, X_train, y_train, n_epoch=10000):
        d = X_train.shape[1]
        if self.P is None:
            self.init_params(d)

        if self.task == 'c' and (np.max(y_train) != 1 or np.min(y_train) != -1):
            raise ValueError('When task is "c", y_train must be {1, -1}^n.')

        print('training...')
        self._fit(X_train, y_train, n_epoch)
        print('training complete.')

    def _fit(self, X_train, y_train, n_epoch):
        n, d = X_train.shape
        if isinstance(self.optimizer, Optimizer):
            opt_method = self._stoc_update
            dptable_anova = np.zeros((self.order+1, d+1))
            dptable_grad = np.zeros((self.order+1, d))
            sum_anova = self._anova(X_train)
            for epoch in range(n_epoch):
                # When a stochastic gradient based method is used, only P[m-2]
                # (i.e., the weights for m-order feature conjunctions) is optimized in each epoch.
                m = (epoch % (self.order-1)) + 2
                opt_method(X_train, y_train, m, dptable_anova, dptable_grad, sum_anova)
                if self.iprint:
                    output = self.b + np.dot(X_train, self.w) + sum_anova
                    self._print(y_train, output, epoch)
        else:
            opt_method = self._coordinate_descent
            dptable_anova = np.zeros((n, self.order+1))
            dptable_grad = np.zeros((n, self.order+1))
            dptable_poly = np.zeros((n, self.order+1))
            output = self.decision_function(X_train)
            idxs = [np.where(X_train[:, i] != 0)[0] for i in range(d)]
            for epoch in range(n_epoch):
                m = (epoch % (self.order-1)) + 2

                output = opt_method(X_train,
                                    y_train,
                                    m,
                                    dptable_anova,
                                    dptable_grad,
                                    dptable_poly,
                                    output,
                                    idxs)
                if self.iprint:
                    self._print(y_train, output, epoch)
