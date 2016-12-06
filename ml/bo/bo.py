import numpy as np
from ..gp import gp
import acquison
from sklearn.cross_validation import KFold
import random
import itertools
import sys


class BO(object):
    def __init__(self, make, eval, intervals, pred=None, grid=1000,
                 fold_num=10, opt_times=100, kernel='Matern52', acq='MI',
                 acqparams=None, candidates=None, values=None, params=None):
        self.make = make
        self.eval = eval
        self.intervals = intervals
        self.pred = pred
        self.grid = grid
        self.fold_num = fold_num
        self.opt_times = opt_times
        self.kernel = kernel
        if acqparams is not None:
            self.acquison = acquison.get_acquison(acq)(acqparams)
        else:
            self.acquison = acquison.get_acquison(acq)()
        self.candidates = candidates
        self.values = values
        self.params = params

    def fit(self, train_x, train_y, valid_x=None, valid_y=None, test_x=None):
        print ('Making candidates ...')

        pred_flag = False
        if self.pred is not None and test_x is not None:
            pred_flag = True
            pred_y = np.zeros((self.opt_times, test_x.shape[0]))

        if hasattr(self.acquison, 'd'):
            self.acquison.d = len(self.intervals)

        if self.candidates is None:
            if type(self.grid) == int:
                grid_list = np.arange(self.grid, dtype=np.float)/self.grid
                cand_list = list(itertools.product(grid_list,
                                                   repeat=len(self.intervals)))
                self.candidates = np.array(cand_list)
            else:
                grid_list = [(np.arange(self.grid[i], dtype=np.float)
                              / self.grid[i])
                             for i in xrange(len(self.intervals))]
                self.candidates = np.array(list(itertools.product(*grid_list)))

        self.intervals = np.array(self.intervals)

        if self.params is None:
            # init hyperparameters and delete in candidates
            next_idx = random.choice(np.arange(self.candidates.shape[0]))
            next = self.candidates[next_idx]
            self.candidates = np.delete(self.candidates, next_idx, 0)
            self.params = np.array(next)

        print ('Optimizing ...')
        for i in xrange(self.opt_times):
            sys.stdout.write('\rIteration:{0}/{1}'.format(i+1, self.opt_times))
            sys.stdout.flush()
            value = 0.0

            # train clf
            # if specify valid data
            if valid_x is not None and valid_y is not None:
                clf = self.make(map_to_origin(next, self.intervals))
                value = self.eval(clf, train_x, train_y, valid_x, valid_y)
                if pred_flag:
                    pred_y[i] = self.pred(clf, test_x)
                del clf
            # else(cross-validation)
            else:
                kf = KFold(train_x.shape[0], n_folds=self.fold_num)
                for train_idx, valid_idx in kf:
                    clf = self.make(map_to_origin(next, self.intervals))
                    value += self.eval(clf, train_x[train_idx],
                                       train_y[train_idx], train_x[valid_idx],
                                       train_y[valid_idx])
                    del clf
                if pred_flag:
                    clf = self.make(map_to_origin(next, self.intervals))
                    dammy = self.eval(clf, train_x, train_y,
                                      train_x[valid_idx], train_y[valid_idx])
                    pred_y[i] = self.pred(clf, test_x)
                    del clf
                value /= (1.0 * self.fold_num)

            # decide next hyperparameters in candidates
            if self.params.ndim == 1:
                self.values = np.array(value)
                next_idx = random.choice(np.arange(self.candidates.shape[0]))
                next = self.candidates[next_idx]
            else:
                self.values = np.append(self.values, value)
                if hasattr(self.acquison, 'best'):
                    self.acquison.best = np.max(self.values)
                gaussian_process = gp.GP(kernel_name=self.kernel, iprint=False)
                gaussian_process.fit(self.params, self.values)
                mean, var = gaussian_process.decision_function(self.candidates)
                del gaussian_process
                next_idx = self.acquison.calc(mean, var)
                next = self.candidates[next_idx]

            # delete next hyperparameters in candidates
            self.candidates = np.delete(self.candidates, next_idx, 0)
            self.params = np.vstack((self.params, next))

        print ('\nOptimizing complete.')

        if not pred_flag:
            return map_to_origin(self.params, self.intervals), self.values
        else:
            return (map_to_origin(self.params, self.intervals),
                    self.values, pred_y)


def map_to_origin(x, intervals):
    return intervals[:, 0] + x * (intervals[:, 1] - intervals[:, 0])
