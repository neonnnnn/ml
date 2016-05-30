import numpy as np
from ..gp import gp
import acquison
from sklearn.cross_validation import KFold
import random
import gc


class BO(object):
    def __init__(self, make, eval, inits, candidates, fold_num=10, opt_times=100, kernel="Matern52", acq="MI", acqparams=None):
        self.make = make
        self.eval = eval
        self.inits = inits
        self.candidates = candidates
        self.fold_num = fold_num
        self.opt_times = opt_times
        self.kernel = kernel
        if acqparams is not None:
            self.acquison = acquison.get_acquison(acq)(acqparams)
        else:
            self.acquison = acquison.get_acquison(acq)()

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        params = np.array(self.inits)
        print ("Optimizing...")
        kf = KFold(train_x.shape[0], n_folds=self.fold_num)
        next = self.inits
        if self.candidates.tolist().count(self.inits) != 0:
            self.candidates = np.delete(self.candidates, self.candidates.tolist().index(self.inits), 0)

        for i in xrange(self.opt_times):
            value = 0.0
            # train clf
            # if specify valid data
            if valid_x is not None and valid_y is not None:
                clf = self.make(next)
                value = self.eval(clf, train_x, train_y, valid_x, valid_y)
                del clf
                gc.collect()
            else:
                for train_idx, valid_idx in kf:
                    clf = self.make(next)
                    value += self.eval(clf, train_x[train_idx], train_y[train_idx], train_x[valid_idx], train_y[valid_idx])
                    del clf
                    gc.collect()
                value /= (1.0 * self.fold_num)

            # decide next hyper-params in candidates
            if i == 0:
                next_idx = random.choice(np.arange(self.candidates.shape[0]))
                next = self.candidates[next_idx]
                values = np.array(value)
            else:
                values = np.append(values, value)
                gaussian_process = gp.GP(kernel_name=self.kernel)
                gaussian_process.fit(params, values)
                mean, var = gaussian_process.decision_function(self.candidates)
                next_idx = self.acquison.calc(mean, var)
                next = self.candidates[next_idx]
                del gaussian_process
                gc.collect()
                self.acquison.gamma = np.delete(self.gamma, next_idx, 0)

            # delete the next hyper-params in candidates
            self.candidates = np.delete(self.candidates, next_idx, 0)

            print values
            print "next:", next
            params = np.vstack((params, next))

        print ("Optimizing complete.")

        return params, values

