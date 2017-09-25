import numpy as np
import kernel
import sys
from scipy import sparse


class SVC(object):
    def __init__(self, C, kernel_name='linear', params=None, eps=1e-4,
                 tau=1e-12, max_iter=100000, wss='WSS3', iprint=True):
        self.C = C
        self.K = kernel.get_kernel(kernel_name, params)
        self.params = params
        self.eps = eps
        self.tau = tau
        self.max_iter = max_iter
        self.WSS = self.__getattribute__(wss)
        self.iprint = iprint
        self.alpha = None
        self.support_vector = None
        self.bias = 0
        self.alpha_times_y = None
        self.cache = {}
        self.cache2 = None
        self.keys = None
        self.flag = False

    def decision_function(self, x):
        output = (self.K.calc(self.support_vector, x)).dot(self.alpha_times_y)
        return output + self.bias

    def predict(self, x):
        return np.sign(self.decision_function(x))

    def score(self, x, y):
        pred = self.predict(x)
        return 1.0 * sum(pred*y > 0) / len(pred)

    def _working_set_selection1(self, grad_f_a, up_idx, low_idx, x, y):
        y_times_grad = y * grad_f_a
        idx1 = up_idx[np.argmax(-y_times_grad[up_idx])]
        idx2 = low_idx[np.argmin(-y_times_grad[low_idx])]
        y1 = y[idx1]
        y2 = y[idx2]
        if not (idx1 in self.cache):
            self.cache[idx1] = y * y1 * (self.K.calc(x, x[idx1])).ravel()

        if not (idx2 in self.cache):
            self.cache[idx2] = y * y2 * (self.K.calc(x, x[idx2])).ravel()

        # convergence test
        if -y_times_grad[idx1]+y_times_grad[idx2] < self.eps:
            self.flag = True

        a = ((self.cache[idx1])[idx1] + (self.cache[idx2])[idx2]
             - 2*y1*y2*(self.cache[idx1])[idx2])
        if a <= 0:
            a = self.tau
        d = (-y_times_grad[idx1]+y_times_grad[idx2]) / a

        return idx1, idx2, d

    def _working_set_selection3(self, grad_f_a, up_idx, low_idx, x, y):
        y_times_grad = y * grad_f_a
        idx1 = up_idx[np.argmax(-y_times_grad[up_idx])]
        t = low_idx[np.where(-y_times_grad[low_idx] < -y_times_grad[idx1])]
        y1 = y[idx1]

        if not (idx1 in self.cache):
            self.cache[idx1] = y * y1 * self.K.calc(x, x[idx1]).ravel()
        Qii = self.cache[idx1]

        newkeys = t[self.keys[t]]
        if len(newkeys) > 0:
            self.keys[newkeys] = False
            self.cache2[newkeys] = self.K.calc_same(x[newkeys])
        Qtt = self.cache2[t]

        ait = Qii[idx1] + Qtt - 2*y1*y[t]*Qii[t]
        ait[np.where(ait <= 0)] = self.tau
        bit = -y_times_grad[idx1] + y_times_grad[t]
        obj_min = -(bit**2) / ait
        t_idx = np.argmin(obj_min)
        idx2 = t[t_idx]

        # convergence test
        if -y_times_grad[idx1]+y_times_grad[idx2] < self.eps:
            self.flag = True

        if not (idx2 in self.cache):
            self.cache[idx2] = y * y[idx2] * self.K.calc(x, x[idx2]).ravel()

        return idx1, idx2, bit[t_idx] / ait[t_idx]

    def check_up_idx(self, y, alpha):
        if ((y == 1) and (alpha < self.C)) or ((y == -1) and (alpha > 0)):
            return True
        else:
            return False

    def check_low_idx(self, y, alpha):
        if ((y == 1) and (alpha > 0)) or ((y == -1) and (alpha < self.C)):
            return True
        else:
            return False

    def init_params(self, y):
        self.alpha = np.zeros(y.shape[0])
        self.cache2 = np.zeros(y.shape[0])
        self.keys = np.ones(y.shape[0], dtype=bool)
        if np.min(y) == 0 and np.max(y) == 1:
            y = 2*y - 1
        return y, y > 0, y < 0, - np.ones(y.shape[0])

    def calc_alpha(self, y1, y2, alpha1_old, alpha2_old, d):
        alpha1_new = alpha1_old + y1*d
        const = y1*alpha1_old + y2*alpha2_old

        if y1 == y2:
            upper = min(self.C, alpha1_old + alpha2_old)
            lower = max(0, alpha1_old + alpha2_old - self.C)
        else:
            upper = min(self.C, alpha1_old - alpha2_old + self.C)
            lower = max(0, alpha1_old - alpha2_old)

        alpha1_new = np.clip(alpha1_new, lower, upper)
        alpha2_new = y2 * (const-y1*alpha1_new)

        return alpha1_new, alpha2_new

    def calc_result(self, x, y, grad_f_a):
        support_vector_idx = np.where(self.alpha > 1e-5*self.C)
        self.support_vector = x[support_vector_idx]
        self.alpha = self.alpha[support_vector_idx]
        self.alpha_times_y = self.alpha * y[support_vector_idx]
        self.bias = np.mean(-y[support_vector_idx] * grad_f_a[support_vector_idx])

    def fit(self, x, y):
        if self.iprint:
            print('training ...')

        if sparse.issparse(x):
            kernel_name = 'Sparse' + self.K.__class__.__name__
            self.K = kernel.get_kernel(kernel_name, self.params)

        # init_params
        y, up_idx, low_idx, grad_f_a = self.init_params(y)

        for i in range(self.max_iter):
            if self.iprint:
                if i % (self.max_iter/1000):
                    sys.stdout.write('\rIteration:{0}/{1}'
                                     .format(i+1, self.max_iter))
                    sys.stdout.flush()
            # select working set
            idx1, idx2, d = self.WSS(grad_f_a, up_idx.nonzero()[0],
                                     low_idx.nonzero()[0], x, y)

            y1 = y[idx1]
            y2 = y[idx2]
            Qii = self.cache[idx1]
            Qjj = self.cache[idx2]
            alpha1_old = self.alpha[idx1]
            alpha2_old = self.alpha[idx2]

            # update alpha_1 and alpha_2
            alpha1_new, alpha2_new = self.calc_alpha(y1, y2, alpha1_old,
                                                     alpha2_old, d)

            # update alpha
            self.alpha[idx1] = alpha1_new
            self.alpha[idx2] = alpha2_new

            # update grad_f_a
            grad_f_a += (Qii * (alpha1_new - alpha1_old)
                         + Qjj * (alpha2_new - alpha2_old))

            # update up_idx
            up_idx[idx1] = self.check_up_idx(y1, alpha1_new)
            up_idx[idx2] = self.check_up_idx(y2, alpha2_new)

            # update low_idx
            low_idx[idx1] = self.check_low_idx(y1, alpha1_new)
            low_idx[idx2] = self.check_low_idx(y2, alpha2_new)

            if self.flag:
                if self.iprint:
                    print('\nConverge.')
                break
        if self.iprint:
            if not self.flag:
                print('')
            print('Training Complete.\nIteration:{0}'.format(i+1))
            print('len(cache):{0}'.format(len(self.cache)))
        self.cache.clear()
        del self.cache2
        self.calc_result(x, y, grad_f_a)

    WSS1 = wss1 = _working_set_selection1
    WSS3 = wss3 = _working_set_selection3
