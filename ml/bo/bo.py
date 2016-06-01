import numpy as np
from ..gp import gp
import acquison
from sklearn.cross_validation import KFold
import random
import itertools
import sys


class BO(object):
    def __init__(self, make, eval, intervals, grid=1000, fold_num=10, opt_times=100, kernel="Matern52", acq="MI", acqparams=None):
        self.make = make
        self.eval = eval
        self.intervals = intervals
        self.grid = grid
        self.fold_num = fold_num
        self.opt_times = opt_times
        self.kernel = kernel
        if acqparams is not None:
            self.acquison = acquison.get_acquison(acq)(acqparams)
        else:
            self.acquison = acquison.get_acquison(acq)()

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        print ("Making candidates ...")

        if hasattr(self.acquison, "d"):
            self.acquison.d = len(self.intervals)

        if type(self.grid) == int:
            grid_list = np.arange(self.grid, dtype=np.float) / self.grid
            candidates = np.array(list(itertools.product(grid_list, repeat=len(self.intervals))))
        else:
            grid_list = [np.arange(self.grid[i], dtype=np.float) / self.grid[i] for i in xrange(len(self.intervals))]
            candidates = np.array(list(itertools.product(*grid_list)))
        self.intervals = np.array(self.intervals)
        next_idx = random.choice(np.arange(candidates.shape[0]))
        next = candidates[next_idx]
        candidates = np.delete(candidates, next_idx, 0)

        params = np.array(next)
        print ("Optimizing ...")
        for i in xrange(self.opt_times):
            sys.stdout.write("\r Iteration:%d/%d" % (i + 1, self.opt_times))
            sys.stdout.flush()
            value = 0.0

            if hasattr(self.acquison, "t"):
                self.acquison.t += 1
            # train clf
            # if specify valid data
            if valid_x is not None and valid_y is not None:
                clf = self.make(map_to_origin(next, self.intervals))
                value = self.eval(clf, train_x, train_y, valid_x, valid_y)
                del clf
            else:
                kf = KFold(train_x.shape[0], n_folds=self.fold_num)
                for train_idx, valid_idx in kf:
                    clf = self.make(map_to_origin(next, self.intervals))
                    value += self.eval(clf, train_x[train_idx], train_y[train_idx], train_x[valid_idx], train_y[valid_idx])
                    del clf
                value /= (1.0 * self.fold_num)

            # decide next hyper-params in candidates
            if i == 0:
                next_idx = random.choice(np.arange(candidates.shape[0]))
                next = candidates[next_idx]
                values = np.array(value)
            else:
                values = np.append(values, value)
                gaussian_process = gp.GP(kernel_name=self.kernel, iprint=False)
                gaussian_process.fit(params, values)
                mean, var = gaussian_process.decision_function(candidates)
                next_idx = self.acquison.calc(mean, var)
                next = candidates[next_idx]
                del gaussian_process
                self.acquison.gamma = np.delete(self.acquison.gamma, next_idx, 0)
            # delete the next hyper-params in candidates
            candidates = np.delete(candidates, next_idx, 0)
            params = np.vstack((params, next))

        print ("Optimizing complete.")

        return map_to_origin(params, self.intervals), values


def map_to_origin(x, intervals):
    return intervals[:, 0] + x * (intervals[:, 1] - intervals[:, 0])


