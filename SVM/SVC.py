import numpy as np
import kernel


class SVC(object):
    def __init__(self, C, Kernel='linear', params=None, eps=1e-3, tau=1e-12, max_iter=1000, wss='WSS1'):
        self.C = C
        self.kernel = kernel.get_kernel(Kernel)
        self.params = params
        self.eps = eps
        self.tau = tau
        self.max_iter = max_iter
        self.alpha = None
        self.bias = 0
        self.y = None
        self.support_vector = None
        self.cache = {}
        self.WSS = self.__getattribute__(wss)

    def working_set_selection1(self, grad_f_a, I_up_idx, I_low_idx, x):
        minus_y_times_grad = - self.y * grad_f_a
        idx1 = I_up_idx[np.argmax(minus_y_times_grad[I_up_idx])]
        idx2 = I_low_idx[np.argmin(minus_y_times_grad[I_low_idx])]
        if not (idx1 in self.cache):
            self.cache[idx1] = self.y * self.y[idx1] * self.kernel(x, x[idx1], self.params).ravel()

        if not (idx2 in self.cache):
            self.cache[idx2] = self.y * self.y[idx2] * self.kernel(x, x[idx2], self.params).ravel()

        if - self.y[idx1] * grad_f_a[idx1] + self.y[idx2] * grad_f_a[idx2] <= self.eps:
            print ("Converge.")
            return -1, -1

        return idx1, idx2

    def working_set_selection3(self, grad_f_a, I_up_idx, I_low_idx, x):
        minus_y_times_grad = - self.y * grad_f_a
        idx1 = I_up_idx[np.argmax(minus_y_times_grad[I_up_idx])]
        t = I_low_idx[np.where(minus_y_times_grad[I_low_idx] < minus_y_times_grad[idx1])].tolist()

        if not (idx1 in self.cache):
            self.cache[idx1] = self.y * self.y[idx1] * self.kernel(x, x[idx1], self.params).ravel()
        Qii = self.cache[idx1]

        Qtt = np.zeros(len(t))
        j = 0
        for i in t:
            if not (i in self.cache):
                self.cache[i] = self.y * self.y[i] * self.kernel(x, x[i], self.params).ravel()
            Qtt[j] = (self.cache[i])[i]
            j += 1

        ait = Qii[idx1] + Qtt - 2 * self.y[idx1] * self.y[t] * Qii[t]
        ait[np.where(ait <= 0)] = self.tau
        bit = minus_y_times_grad[idx1] - minus_y_times_grad[t]
        obj_min = - (bit ** 2) / ait

        if minus_y_times_grad[idx1] - np.min(obj_min) <= self.eps:
            print ("Converge.")
            return -1, -1

        idx2 = t[np.argmin(obj_min)]
        if not (idx2 in self.cache):
            self.cache[idx2] = self.y * self.y[idx2] * self.kernel(x, x[idx2], self.params)

        return idx1, idx2

    def check_I_up_idx(self, y, alpha):
        if ((y == 1) and (alpha < self.C)) or ((y == -1) and (alpha > 0)):
            return True
        else:
            return False

    def check_I_low_idx(self, y, alpha):
        if ((y == 1) and (alpha > 0)) or ((y == -1) and (alpha < self.C)):
            return True
        else:
            return False

    def decision_function(self, x):
        return np.dot(self.kernel(self.support_vector, x, self.params), self.alpha * self.y) + self.bias

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

        for i in xrange(self.max_iter):
            # select working set
            idx1, idx2 = self.WSS(grad_f_a, I_up_idx.nonzero()[0], I_low_idx.nonzero()[0], x)
            if idx1 == -1:
                break
            if i == 20000:
                self.WSS = self.__getattribute__("WSS3")
            y1 = y[idx1]
            y2 = y[idx2]
            Qii = self.cache[idx1]
            Qjj = self.cache[idx2]

            alpha1_old = self.alpha[idx1]
            alpha2_old = self.alpha[idx2]

            # update alpha_1 and alpha_2
            a = Qii[idx1] + Qjj[idx2] - 2 * y1 * y2 * Qii[idx2]
            if a <= 0:
                a = self.tau
            d = (- y1 * grad_f_a[idx1] + y2 * grad_f_a[idx2]) / a
            alpha1_new = alpha1_old + y1 * d
            sum = y1 * alpha1_old + y2 * alpha2_old
            if alpha1_new > self.C:
                alpha1_new = self.C
            elif alpha1_new < 0:
                alpha1_new = 0

            alpha2_new = y2 * (sum - y1 * alpha1_new)
            if alpha2_new > self.C:
                alpha2_new = self.C
            elif alpha2_new < 0:
                alpha2_new = 0

            alpha1_new = y1 * (sum - y2 * alpha2_new)
            self.alpha[idx2] = alpha2_new
            self.alpha[idx1] = alpha1_new

            # update grad_f_a
            grad_f_a += Qii * (alpha1_new - alpha1_old) + Qjj * (alpha2_new - alpha2_old)

            # update I_up_idx
            I_up_idx[idx1] = self.check_I_up_idx(y1, alpha1_new)
            I_up_idx[idx2] = self.check_I_up_idx(y2, alpha2_new)

            # update I_low_idx
            I_low_idx[idx1] = self.check_I_low_idx(y1, alpha1_new)
            I_low_idx[idx2] = self.check_I_low_idx(y2, alpha2_new)

        print ('Training Complete. Iteration:'), (i)

        support_vector_idx = np.where(self.alpha > 1e-5 * self.C)
        self.support_vector = x[support_vector_idx]
        self.alpha = self.alpha[support_vector_idx]
        self.y = self.y[support_vector_idx]
        self.bias = np.average(self.y - np.dot(self.alpha * self.y, self.kernel(self.support_vector, self.support_vector, self.params)))

    WSS1 = wss1 = working_set_selection1
    WSS3 = wss3 = working_set_selection3





