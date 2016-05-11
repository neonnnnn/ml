from svc import SVC
import numpy as np
import sparsekernel
import sys


class SparseSVC(SVC):
    def __init__(self,  C, kernel_name='linear', params=None, eps=1e-3, tau=1e-12, max_iter=10000, wss='WSS1'):
        self.C = C
        self.K = sparsekernel.get_kernel(kernel_name)(params)
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
        self.cache2 = None
        self.keys = None
        self.flag = False

    def _working_set_selection1(self, grad_f_a, up_idx, low_idx, x, y):
        minus_y_times_grad = - y * grad_f_a
        idx1 = up_idx[np.argmax(minus_y_times_grad[up_idx])]
        idx2 = low_idx[np.argmin(minus_y_times_grad[low_idx])]
        if not (idx1 in self.cache):
            self.cache[idx1] = self.K.calc_kernel(x, x[idx1])

        if not (idx2 in self.cache):
            self.cache[idx2] = self.K.calc_kernel(x, x[idx2])

        if minus_y_times_grad[idx1] - minus_y_times_grad[idx2] < self.eps:
            self.flag = True

        a = (self.cache[idx1])[0, idx1] + (self.cache[idx2])[0, idx2] - 2 * (self.cache[idx1])[0, idx2]
        if a <= 0:
            a = self.tau
        d = (minus_y_times_grad[idx1] - minus_y_times_grad[idx2]) / a

        return idx1, idx2, d

    def _working_set_selection3(self, grad_f_a, up_idx, low_idx, x, y):
        minus_y_times_grad = - y * grad_f_a
        idx1 = up_idx[np.argmax(minus_y_times_grad[up_idx])]
        t = low_idx[np.where(minus_y_times_grad[low_idx] < minus_y_times_grad[idx1])]

        if not (idx1 in self.cache):
            self.cache[idx1] = self.K.calc_kernel(x, x[idx1])
        Qii = self.cache[idx1]

        newkeys = t[self.keys[t]]
        if len(newkeys) > 0:
            self.keys[newkeys] = False
            self.cache2[newkeys] = self.K.calc_kernel_same(x[newkeys])
        Qtt = self.cache2[t]

        ait = Qii[0, idx1] + Qtt - 2 * Qii[0, t].toarray().ravel()
        ait[np.where(ait <= 0)] = self.tau
        bit = minus_y_times_grad[idx1] - minus_y_times_grad[t]
        obj_min = - (bit ** 2) / ait
        t_idx = np.argmin(obj_min)
        idx2 = t[t_idx]

        if minus_y_times_grad[idx1] - minus_y_times_grad[idx2] < self.eps:
            self.flag = True

        if not (idx2 in self.cache):
            self.cache[idx2] = self.K.calc_kernel(x, x[idx2])

        return idx1, idx2, bit[t_idx] / ait[t_idx]

    def fit(self, x, y):
        print ("training ...")

        # init_params
        y, up_idx, low_idx, grad_f_a = super(SparseSVC, self).init_params(y)
        self.cache2 = np.zeros(x.shape[0])
        self.keys = np.ones(x.shape[0], dtype=bool)

        for i in range(self.max_iter):
            if i % (self.max_iter / 1000):
                sys.stdout.write("\r Iteration:%d/%d" % (i, self.max_iter))
                sys.stdout.flush()

            # select working set
            idx1, idx2, d = self.WSS(grad_f_a, up_idx.nonzero()[0], low_idx.nonzero()[0], x, y)

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
            grad_f_a += np.array(Qii.multiply(y1 * y) * (alpha1_new - alpha1_old) + Qjj.multiply(y2 * y) * (alpha2_new - alpha2_old)).ravel()

            up_idx[idx1] = self.check_up_idx(y1, alpha1_new)
            up_idx[idx2] = self.check_up_idx(y2, alpha2_new)

            # update I_low_idx
            low_idx[idx1] = self.check_low_idx(y1, alpha1_new)
            low_idx[idx2] = self.check_low_idx(y2, alpha2_new)

            if self.flag:
                print ("\nConverge.")
                break
        if not self.flag:
            print ("")
        print ('Training Complete. \nIteration:'), (i+1)
        print ('len(cache):'), len(self.cache)
        self.cache.clear()
        del self.cache2
        self.calc_result(x, y, grad_f_a)

    WSS1 = wss1 = _working_set_selection1
    WSS3 = wss3 = _working_set_selection3

