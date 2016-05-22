import numpy as np
import kernel
import scipy.optimize as spopt


class GP(object):
    def __init__(self, kernel_name="Matern52", alpha=None, beta=None, theta=None, type2ml=True):
        self.K = kernel.get_kernel(kernel_name)(theta)
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.type2ml = type2ml
        self.t = None
        self.x = None
        self.C = None

    def negative_log_likelihood(self, params):
        alpha = np.exp(params[0])
        beta = np.exp(params[1])
        theta = np.exp(params[2:])
        C = self.K.calc_kernel(self.x, self.x, theta) * alpha + np.identity(self.x.shape[0]) * beta
        inv_C_times_t = np.linalg.solve(C, self.t)
        return 0.5 * np.log(np.linalg.det(C) + 1e-30) + 0.5 * np.dot(inv_C_times_t, self.t) + 0.5 * self.t.shape[0] * np.log(2 * np.pi)

    def grad_nll(self, params):
        alpha = np.exp(params[0])
        beta = np.exp(params[1])
        theta = np.exp(params[2:])

        K = self.K.calc_kernel(self.x, self.x, theta)
        C = alpha * K + beta * np.identity(self.x.shape[0])
        inv_C = np.linalg.inv(C)
        t_times_inv_C = np.dot(inv_C, self.t)
        grad_C_alpha = K * alpha
        grad_C_theta = self.K.calc_grad(self.x, self.x, theta) * alpha

        grad = np.zeros(len(params))
        grad[0] = 0.5 * np.trace(np.dot(inv_C, grad_C_alpha)) - 0.5 * np.dot(np.dot(t_times_inv_C, grad_C_alpha), t_times_inv_C)
        grad[1] = 0.5 * np.trace(inv_C) * beta - 0.5 * np.dot(t_times_inv_C, t_times_inv_C) * beta

        for i in xrange(self.K.dim):
            grad[i+2] = 0.5 * np.trace(np.dot(inv_C, grad_C_theta[i])) - 0.5 * np.dot(np.dot(t_times_inv_C, grad_C_theta[i]), t_times_inv_C)

        return grad

    def fit(self, x, y, hyper_opt_iter=30):
        print ("training...")
        if len(x.shape) == 1:
            dim = 1
            self.x = np.atleast_2d(x).T
        else:
            self.x = x
            dim = self.x.shape[1]
        self.t = y

        if self.K.ard:
            self.K.dim += dim

        if self.type2ml:
            print (" Doing Type 2 Maximum Likelihood...")
            f_min = np.inf
            for i in xrange(hyper_opt_iter):
                init_params = np.random.rand(self.K.dim + 2)
                res = spopt.fmin_cg(self.negative_log_likelihood, init_params, fprime=self.grad_nll, full_output=True, disp=False)

                if res[1] < f_min:
                    f_min = res[1]
                    params = res[0]

            self.alpha = np.exp(params[0])
            self.beta = np.exp(params[1])
            self.theta = np.exp(params[2:])

        # number of train data = N : x.shape[0]
        # C.shape : (N, N)
        self.C = self.alpha * self.K.calc_kernel(self.x, self.x, self.theta) + self.beta * np.identity(self.x.shape[0])

    def decision_function(self, x):
        if len(x.shape) == 1:
            x = np.atleast_2d(x).T
        # number of train data = N = self.x.shape[0]
        # number of test data = M = x.shape[0]
        # k.shape = (M,N)
        k = self.K.calc_kernel(self.x, x, self.theta) * self.alpha
        # c.shape = (M,)
        if len(x.shape) == 1:
            c = self.alpha * self.K.calc_kernel_same(x) + 1.0 * self.beta
        else:
            c = self.alpha * self.K.calc_kernel_diag(x) + 1.0 * self.beta

        # Trans_k_dot_inv_C.shape = (M,N)
        Trans_k_dot_inv_C = np.linalg.solve(self.C, k.T).T

        # mean.shape = (M, )
        mean = np.dot(Trans_k_dot_inv_C, self.t)
        # var.shape = (M,)
        var = - np.sum(Trans_k_dot_inv_C * k, axis=1) + c
        return mean, var

    def predict(self, x):
        m, var = self.decision_function(x)
        return m

