import numpy as np
import kernel
import sys
import svc
import scipy.sparse as sp


class TSVM(object):
    def __init__(self, C_unlabel, s=0., max_iter_tsvm=100, C=1.0, kernel_name='linear', params=None, eps=1e-4, tau=1e-12, max_iter=10000, wss='WSS3',
                 sparse=False, iprint=True):
        self.C_unlabel = C_unlabel
        self.s = s
        if self.s > 0:
            self.s = 0.
        self.max_iter2 = max_iter_tsvm

        self.C = C
        if sparse:
            self.K = kernel.get_kernel("Sparse" + kernel_name)(params)
        else:
            self.K = kernel.get_kernel(kernel_name)(params)
        self.params = params
        self.eps = eps
        self.tau = tau
        self.max_iter = max_iter
        self.iprint = iprint
        self.alpha = None
        self.support_vector = None
        self.bias = 0
        self.alpha_times_y = None
        self.cache = {}
        self.cache2 = None
        self.keys = None
        self.flag = False
        self.clf = svc.SVC(C, kernel_name, params, eps, tau, max_iter, wss, sparse, iprint=False)
        self.L = None
        self.U = None

    def decision_function(self, x):
        return (self.K.calc_kernel(self.support_vector, x)).dot(self.alpha_times_y) + self.bias

    def predict(self, x):
        return np.sign(self.decision_function(x))

    def score(self, x, y):
        pred = self.predict(x)
        return 1.0 * sum(pred * y > 0) / len(pred)

    def working_set_selection(self, x, y, grad, up_idx, low_idx, zeta):
        idx1 = up_idx[np.argmax(grad[up_idx])]
        idx2 = low_idx[np.argmin(grad[low_idx])]
        y1 = y[idx1]
        y2 = y[idx2]
        if not (idx1 in self.cache):
            if idx1 != 0:
                K1 = (self.K.calc_kernel(x, x[idx1])).ravel()
                self.cache[idx1] = np.append(np.mean(K1[self.L:]), K1)
            else:
                K0 = self.K.calc_kernel(x[:self.L + self.U], x[self.L:self.L+self.U])
                K0 = np.mean(K0, axis=1)
                K0 = np.append(K0, K0[self.L:])
                self.cache[idx1] = np.append(np.mean(K0[self.L:]), K0)
        if not (idx2 in self.cache):
            if idx1 != 0:
                K2 = (self.K.calc_kernel(x, x[idx2])).ravel()
                self.cache[idx2] = np.append(np.mean(K2[self.L:]), K2)
            else:
                K0 = self.K.calc_kernel(x[:self.L + self.U], x[self.L:self.L+self.U])
                K0 = np.mean(K0, axis=1)
                K0 = np.append(K0, K0[self.L:])
                self.cache[idx1] = np.append(np.mean(K0[self.L:]), K0)

        # convergence test
        if grad[idx1] - grad[idx2] < self.eps:
            self.flag = True

        a = (self.cache[idx1])[idx1] + (self.cache[idx2])[idx2] - 2 * y1 * y2 * (self.cache[idx1])[idx2]
        if a <= 0:
            a = self.tau
        d = (-y1 * np.sum(self.cache[idx1] * y * self.alpha) + zeta[idx1] + y2 * np.sum(self.cahce[idx2] * y * self.alpha) - zeta[idx2]) / a

        if np.abs(d) < self.eps:
            self.flag = True

        return idx1, idx2, d

    def check_up_idx(self, y, alpha, beta):
        up_idx = ((y * alpha + beta) >= 0) + (((y * alpha + beta) == self.C) * (y < 0))
        return up_idx

    def check_low_idx(self, y, alpha, beta):
        low_idx = ((y * alpha + beta) > 0) + (y > 0) * (beta == self.C_unlabel)
        return low_idx

    def calc_beta(self, x_unlabeled):
        beta = np.zeros(self.L + 2 * self.U + 1)
        f_u = self.decision_function(x_unlabeled)
        y_times_f_u = np.append(f_u, -f_u)
        (beta[self.L + 1:])[np.where(y_times_f_u < self.s)] = self.C_unlabel

        return beta

    def init_params(self, x_labeled, x_unlabeled, y):
        self.L = y.shape[0]
        self.U = x_unlabeled.shape[0]

        if self.iprint:
            print ("training SVM ...")
        self.clf.fit(x_labeled, y)
        if self.iprint:
            print ("training SVM complete")

        self.alpha_times_y = self.clf.alpha_times_y
        self.support_vector = self.clf.support_vector
        self.bias = self.clf.bias
        self.alpha = self.clf.alpha

        self.C = np.zeros(self.L + 2 * self.U + 1)
        self.C[1:self.L + 1] = self.clf.C
        self.C[self.L + 1:] = self.C_unlabel
        if self.sparse:
            x = sp.vstack((x_labeled, x_unlabeled))
        else:
            x = np.vstack((x_labeled, x_unlabeled))
        y_all = np.append(1, y)
        y_all = np.append(y_all, np.ones(self.U))
        y_all = np.append(y_all, -np.ones(self.U))

        return x, y_all

    def calc_result(self, x, y, beta, grad, zeta):
        support_vector_idx = np.where(self.alpha * y + beta > 1e-5 * self.C)
        self.support_vector = x[support_vector_idx]
        self.alpha_times_y = self.alpha[support_vector_idx] * y[support_vector_idx]
        b_idx = np.where(self.alpha * y + beta > 1e-5 * self.C) * np.where(self.alpha * y + beta < self.C)
        self.bias = np.average(np.sum(b_idx) - (grad[b_idx] + zeta[b_idx]))

    def calc_alpha(self, y1, y2, alpha1_old, alpha2_old, idx1, idx2, beta, d):
        const = alpha1_old + alpha2_old
        alpha1_new = alpha1_old + d

        if alpha1_new * y1 + beta[idx1] > self.C[idx1]:
            alpha1_new = (self.C[idx1] - beta[idx1]) / y1
        elif alpha1_new * y1 + beta[idx1] < 0:
            alpha1_new = -beta[idx1] / y1

        alpha2_new = const - alpha1_new
        if alpha2_new * y2 + beta[idx2] > self.C[idx2]:
            alpha2_new = (self.C[idx2] - beta[idx2]) / y2
            alpha1_new = const - alpha2_new
        elif alpha2_new * y2 + beta[idx2] < 0:
            alpha2_new = - beta[idx2] / y2
            alpha1_new = const - alpha2_new

        return alpha1_new, alpha2_new

    def fit(self, x_labeled, y, x_unlabeled):
        x, y_all = self.init_params(x_labeled, x_unlabeled, y)
        beta = self.calc_beta(x_unlabeled)
        self.alpha = np.zeros(self.L + self.U * 2 + 1)
        zeta = np.array(y_all)
        zeta[0] = np.mean(y_all[1:self.L + 1])
        grad = -zeta

        if self.iprint:
            print ("training TSVM ...")

        for j in range(self.max_iter2):
            if self.iprint:
                sys.stdout.write("\r Iteration:%d/%d" % (i+1, self.max_iter2))
                sys.stdout.flush()

            up_idx = self.check_up_idx(y_all, self.alpha, beta)
            low_idx = self.check_low_idx(y_all, self.alpha, beta)
            up_idx[0] = True
            low_idx[0] = False

            for i in xrange(self.max_iter):
                idx1, idx2, d = self.working_set_selection(x, y_all, grad, up_idx, low_idx, zeta)

                y1 = y_all[idx1]
                y2 = y_all[idx2]

                alpha1_old = self.alpha[idx1]
                alpha2_old = self.alpha[idx2]

                Kii = self.cache[idx1]
                Kjj = self.cache[idx2]

                # update alpha_1 and alpha_2
                alpha1_new, alpha2_new = self.calc_alpha(y1, y2, alpha1_old, alpha2_old, idx1, idx2, beta, d)

                # update alpha
                self.alpha[idx1] = alpha1_new
                self.alpha[idx2] = alpha2_new

                # update grad
                grad += Kii * (alpha1_new - alpha1_old) + Kjj * (alpha2_new - alpha2_old)

                # update up_idx and low_idx
                if idx1 != 0:
                    up_idx[idx1] = self.check_up_idx(y1, alpha1_new, beta[idx1])
                    low_idx[idx1] = self.check_low_idx(y1, alpha1_new, beta[idx1])
                if idx2 != 0:
                    up_idx[idx2] = self.check_low_idx(y2, alpha2_new, beta[idx2])
                    low_idx[idx2] = self.check_low_idx(y2, alpha2_new, beta[idx2])

                if self.flag:
                    break

            self.calc_result(x, y_all, grad, beta, zeta)
            beta_new = self.calc_beta(x_unlabeled)
            if np.allclose(beta, beta_new):
                break
            beta = beta_new


