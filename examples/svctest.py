import numpy as np
from ml.bo import bo
import load_mnist
import itertools
from ml.svm import svc
if __name__ == '__main__':
    dataset = load_mnist.load_data()
    x_train, y_train = dataset[0]
    x_test, y_test = dataset[1]

    x_train = x_train[np.where((y_train == 5) | (y_train == 8))]
    y_train = y_train[np.where((y_train == 5) | (y_train == 8))]
    y_train = ((y_train - 5) / 3) * 2 - 1

    x_test = x_test[np.where((y_test == 5) | (y_test == 8))]
    y_test = y_test[np.where((y_test == 5) | (y_test == 8))]
    y_test = ((y_test - 5) / 3) * 2 - 1

    # Set parameters

    def make(params):
        clf = svc.SVC(C=params[0], params=[params[1]], kernel_name='rbf', max_iter=10000, wss='wss3')
        return clf

    def eval(clf, train_x, train_y, valid_x, valid_y):
        clf.fit(train_x, train_y)
        score = clf.score(valid_x, valid_y)
        return score

    inits = [1, 0.5]
    candidate1 = np.linspace(0.5, 10.4, 100)
    candidate2 = np.linspace(0.05, 2.04, 200)
    candidates = np.array(list(itertools.product(candidate1, candidate2)))

    opt = bo.BO(make=make, eval=eval, inits=inits, candidates=candidates, fold_num=5)
    #params, values = opt.fit(x_train, y_train)
    params, values = opt.fit(x_train, y_train, x_test, y_test)
    np.savetxt("bo_svm_params.txt", params)
    np.savetxt("bo_svm_values.txt", values)

