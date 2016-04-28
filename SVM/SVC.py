import numpy as np
import kernel


class SVC(object):
    def __init__(self, C, Kernel='linear', params=None, eps=1e-3, tau=1e-12, max_iter=1000, wss='WSS1'):
        self.C = C
        self.K = kernel.get_kernel(Kernel)(params)
        self.params = params
        self.eps = eps
        self.tau = tau
        self.max_iter = max_iter
        self.alpha = None
        self.bias = 0
        self.y = None
        self.support_vector = None
        self.cache = {}
        self.cache2 = {}
        self.WSS = self.__getattribute__(wss)

    def working_set_selection1(self, grad_f_a, I_up_idx, I_low_idx, x):
        minus_y_times_grad = - self.y * grad_f_a
        idx1 = I_up_idx[np.argmax(minus_y_times_grad[I_up_idx])]
        idx2 = I_low_idx[np.argmin(minus_y_times_grad[I_low_idx])]
        y1 = self.y[idx1]
        y2 = self.y[idx2]
        if not (idx1 in self.cache):
            self.cache[idx1] = self.y * y1 * self.K.calc_kernel(x, x[idx1]).ravel()

        if not (idx2 in self.cache):

            self.cache[idx2] = self.y * y2 * self.K.calc_kernel(x, x[idx2]).ravel()

        if - self.y[idx1] * grad_f_a[idx1] + self.y[idx2] * grad_f_a[idx2] <= self.eps:
            print ("Converge.")
            return -1, -1, -1

        a = (self.cache[idx1])[idx1] + (self.cache[idx2])[idx2] - 2 * self.y[idx1] * self.y[idx2] * (self.cache[idx1])[idx2]
        if a <= 0:
            a = self.tau
        d = (minus_y_times_grad[idx1] - minus_y_times_grad[idx2]) / a

        return idx1, idx2, d

    def working_set_selection3(self, grad_f_a, I_up_idx, I_low_idx, x):
        minus_y_times_grad = - self.y * grad_f_a
        idx1 = I_up_idx[np.argmax(minus_y_times_grad[I_up_idx])]
        t = I_low_idx[np.where(minus_y_times_grad[I_low_idx] < minus_y_times_grad[idx1])].tolist()

        if not (idx1 in self.cache):
            self.cache[idx1] = self.y * self.y[idx1] * self.K.calc_kernel(x, x[idx1]).ravel()
            self.cache2[idx1] = (self.cache[idx1])[idx1]

        Qii = self.cache[idx1]

        Qtt = np.ones(len(t))
        j = 0

        for i in t:
            if not (i in self.cache2):
                self.cache2[i] = self.K.calc_kernel_diagonal(x[i])
            Qtt[j] = self.cache2[i]
            j += 1

        ait = Qii[idx1] + Qtt - 2 * self.y[idx1] * self.y[t] * Qii[t]
        ait[np.where(ait <= 0)] = self.tau
        bit = minus_y_times_grad[idx1] - minus_y_times_grad[t]
        obj_min = - (bit ** 2) / ait

        if minus_y_times_grad[idx1] - np.min(obj_min) <= self.eps:
            print ("Converge.")
            return -1, -1, -1

        t_idx = np.argmin(obj_min)
        idx2 = t[t_idx]
        if not (idx2 in self.cache):
            self.cache[idx2] = self.y * self.y[idx2] * self.K.calc_kernel(x, x[idx2]).ravel()

        return idx1, idx2, bit[t_idx]/ait[t_idx]

    def decision_function(self, x):
        return np.dot(self.K.calc_kernel(self.support_vector, x), self.alpha * self.y) + self.bias

    def predict(self, x):
        return np.sign(self.decision_function(x))

    def score(self, x, y):
        pred = self.predict(x)
        if len(pred) == 0:
            return 0.0
        return 1.0 * sum(pred * y > 0) / len(pred)

    def fit(self, x, y):
        print ("training ...")
        if np.min(y) == 0 and np.max(y) == 1:
            y = 2*y - 1
        self.y = y
        self.alpha = np.zeros(y.shape[0])
        self.bias = 0

        I_up_idx = self.y > 0
        I_low_idx = self.y < 0

        grad_f_a = - np.ones(y.shape[0])

        for i in range(self.max_iter):
            # select working set
            idx1, idx2, d = self.WSS(grad_f_a, I_up_idx.nonzero()[0], I_low_idx.nonzero()[0], x)
            if idx1 == -1:
                break
            y1 = self.y[idx1]
            y2 = self.y[idx2]
            Qii = self.cache[idx1]
            Qjj = self.cache[idx2]

            alpha1_old = self.alpha[idx1]
            alpha2_old = self.alpha[idx2]

            # update alpha_1 and alpha_2
            alpha1_new = alpha1_old + y1 * d
            sum = y1 * alpha1_old + y2 * alpha2_old

            if alpha1_new > self.C:
                alpha1_new = self.C

            elif alpha1_new < 0:
                alpha1_new = 0

            alpha2_new = y2 * (sum - y1 * alpha1_new)

            if alpha2_new > self.C:
                alpha2_new = self.C
                alpha1_new = y1 * (sum - y2 * alpha2_new)
            elif alpha2_new < 0:
                alpha2_new = 0
                alpha1_new = y1 * sum

            self.alpha[idx2] = alpha2_new
            self.alpha[idx1] = alpha1_new

            # update grad_f_a
            grad_f_a += Qii * (alpha1_new - alpha1_old) + Qjj * (alpha2_new - alpha2_old)

            # update I_up_idx
            if not (alpha1_new < self.C):
                I_up_idx[idx1] = False
            I_up_idx[idx2] = True

            # update I_low_idx
            I_low_idx[idx1] = True
            if not (alpha2_new < self.C):
                I_low_idx[idx2] = False

        print ('Training Complete. Iteration:'), (i)

        support_vector_idx = np.where(self.alpha > 1e-5 * self.C)
        self.support_vector = x[support_vector_idx]
        self.alpha = self.alpha[support_vector_idx]
        self.y = self.y[support_vector_idx]
        self.bias = np.average(self.y - np.dot(self.alpha * self.y, self.K.calc_kernel(self.support_vector, self.support_vector)))

    WSS1 = wss1 = working_set_selection1
    WSS3 = wss3 = working_set_selection3
