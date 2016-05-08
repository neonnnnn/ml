import numpy as np
import kernel
from operator import itemgetter


class SVC(object):
    def __init__(self, C, kernel_name='linear', params=None, eps=1e-3, tau=1e-12, max_iter=10000, wss='WSS1', sparse=False):
        self.C = C
        if sparse:
            self.K = kernel.get_kernel("Sparse" + kernel_name)(params)
        else:
            self.K = kernel.get_kernel(kernel_name)(params)
        self.params = params
        self.eps = eps
        self.tau = tau
        self.max_iter = max_iter
        self.WSS = self.__getattribute__(wss)
        self.alpha = None
        self.support_vector = None
        self.bias = 0
        self.alpha_times_y = None
        self.cache = {}
        self.cache2 = {}

    def decision_function(self, x):
        return (self.K.calc_kernel(self.support_vector, x)).dot(self.alpha_times_y) + self.bias

    def predict(self, x):
        return np.sign(self.decision_function(x))

    def score(self, x, y):
        pred = self.predict(x)
        return 1.0 * sum(pred * y > 0) / len(pred)

    def _working_set_selection1(self, grad_f_a, up_idx, low_idx, x, y):
        minus_y_times_grad = - y * grad_f_a
        idx1 = up_idx[np.argmax(minus_y_times_grad[up_idx])]
        idx2 = low_idx[np.argmin(minus_y_times_grad[low_idx])]
        y1 = y[idx1]
        y2 = y[idx2]
        if not (idx1 in self.cache):
            self.cache[idx1] = y * y1 * (self.K.calc_kernel(x, x[idx1])).ravel()

        if not (idx2 in self.cache):
            self.cache[idx2] = y * y2 * (self.K.calc_kernel(x, x[idx2])).ravel()

        if minus_y_times_grad[idx1] - minus_y_times_grad[idx2] <= self.eps:
            print ("Converge.")
            return idx1, idx2, None

        a = (self.cache[idx1])[idx1] + (self.cache[idx2])[idx2] - 2 * y1 * y2 * (self.cache[idx1])[idx2]
        if a <= 0:
            a = self.tau
        d = (minus_y_times_grad[idx1] - minus_y_times_grad[idx2]) / a

        return idx1, idx2, d

    def _working_set_selection3(self, grad_f_a, up_idx, low_idx, x, y):
        minus_y_times_grad = - y * grad_f_a
        idx1 = up_idx[np.argmax(minus_y_times_grad[up_idx])]
        t = low_idx[np.where(minus_y_times_grad[low_idx] < minus_y_times_grad[idx1])].tolist()
        y1 = y[idx1]

        if not (idx1 in self.cache):
            self.cache[idx1] = y * y1 * self.K.calc_kernel(x, x[idx1]).ravel()
        Qii = self.cache[idx1]

        newkeys = set(t).difference(self.cache2.keys())
        for i in newkeys:
            self.cache2[i] = self.K.calc_kernel_same(x[i])
        Qtt = np.array(itemgetter(*t)(self.cache2))

        ait = Qii[idx1] + Qtt - 2 * y1 * y[t] * Qii[t]
        ait[np.where(ait <= 0)] = self.tau
        bit = minus_y_times_grad[idx1] - minus_y_times_grad[t]
        obj_min = - (bit ** 2) / ait
        t_idx = np.argmin(obj_min)
        idx2 = t[t_idx]

        if minus_y_times_grad[idx1] - obj_min[t_idx] <= self.eps:
            print ("Converge.")
            return idx1, idx2, None

        if not (idx2 in self.cache):
            self.cache[idx2] = y * y[idx2] * self.K.calc_kernel(x, x[idx2]).ravel()

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

    @staticmethod
    def init_params(y):
        if np.min(y) == 0 and np.max(y) == 1:
            y = 2*y - 1
        return y, y > 0, y < 0, - np.ones(y.shape[0]), np.zeros(y.shape[0])

    def calc_alpha(self, y1, y2, alpha1_old, alpha2_old, d):
        alpha1_new = alpha1_old + y1 * d
        const = y1 * alpha1_old + y2 * alpha2_old

        if alpha1_new > self.C:
            alpha1_new = self.C
        elif alpha1_new < 0:
            alpha1_new = 0

        alpha2_new = y2 * (const - y1 * alpha1_new)

        if alpha2_new > self.C:
            alpha2_new = self.C
            alpha1_new = y1 * (const - y2 * alpha2_new)
        elif alpha2_new < 0:
            alpha2_new = 0
            alpha1_new = y1 * const

        return alpha1_new, alpha2_new

    def calc_result(self, x, y, grad_f_a):
        support_vector_idx = np.where(self.alpha > 1e-5 * self.C)
        self.support_vector = x[support_vector_idx]
        self.alpha = self.alpha[support_vector_idx]
        self.alpha_times_y = self.alpha * y[support_vector_idx]
        self.bias = np.average(-y[support_vector_idx] * grad_f_a[support_vector_idx])

    def fit(self, x, y):
        print ("training ...")

        # init_params
        y, up_idx, low_idx, grad_f_a, self.alpha = self.init_params(y)

        for i in range(self.max_iter):
            # select working set
            idx1, idx2, d = self.WSS(grad_f_a, up_idx.nonzero()[0], low_idx.nonzero()[0], x, y)
            # if converge
            if d is None:
                break

            y1 = y[idx1]
            y2 = y[idx2]
            Qii = self.cache[idx1]
            Qjj = self.cache[idx2]
            alpha1_old = self.alpha[idx1]
            alpha2_old = self.alpha[idx2]

            # update alpha_1 and alpha_2
            alpha1_new, alpha2_new = self.calc_alpha(y1, y2, alpha1_old, alpha2_old, d)

            # update alpha
            self.alpha[idx1] = alpha1_new
            self.alpha[idx2] = alpha2_new

            # update grad_f_a
            grad_f_a += Qii * (alpha1_new - alpha1_old) + Qjj * (alpha2_new - alpha2_old)

            # update up_idx
            up_idx[idx1] = self.check_up_idx(y1, alpha1_new)
            up_idx[idx2] = self.check_up_idx(y2, alpha2_new)

            # update low_idx
            low_idx[idx1] = self.check_low_idx(y1, alpha1_new)
            low_idx[idx2] = self.check_low_idx(y2, alpha2_new)

        print ('Training Complete. Iteration:'), (i)
        print ('len(cache):'), len(self.cache)
        self.cache.clear()
        self.cache2.clear()
        self.calc_result(x, y, grad_f_a)

    WSS1 = wss1 = _working_set_selection1
    WSS3 = wss3 = _working_set_selection3
