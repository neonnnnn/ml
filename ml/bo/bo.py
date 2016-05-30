import numpy as np
from ..gp import gp
import acquison
from sklearn.cross_validation import KFold


class BO(object):
    def __init__(self, make, eval, inits, candidates, fold_num=10, opt_times=100, kernel="Matern52", acq="MI", acqparams=None):
        self.make = make
        self.eval = eval
        self.inits = inits
        self.candidates = candidates
        self.fold_num = fold_num
        self.opt_times = opt_times
        self.gaussian_process = gp.GP(kernel_name=kernel)
        if acqparams is not None:
            self.acquison = acquison.get_acquison("acquison")(acqparams)
        else:
            self.acquison = acquison.get_acquison("acquison")()

    def fit(self, x, y):
        clf = self.make(self.inits)
        evals = np.array(self.inits)
        values = []
        print ("Optimizing...")
        for i in xrange(self.opt_times):
            value = 0.0
            kf = KFold(x.shape[0], n_folds=self.fold_num)
            for train_idx, valid_idx in kf:
                value += self.eval(clf, x[train_idx], y[train_idx], x[valid_idx], y[valid_idx])
            value /= self.fold_num
            values += [value]

            self.gaussian_process.fit(evals, np.array(values))
            mean, var = self.gaussian_process.decision_function(self.candidates)
            next_idx = self.acquison.calc(mean, var)
            next = self.candidates[next_idx]
            clf = self.make(next)
            evals = np.vstack((evals, next))
