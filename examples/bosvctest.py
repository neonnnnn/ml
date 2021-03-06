import numpy as np
from ml.bo import bo
import load_mnist
import time
from ml.svm import svc
from sklearn import svm


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
    print('train:', len(y_train))
    print('test:', len(y_test))
    """
    # Set parameters
    def make(params):
        clf = svc.SVC(C=params[0], params=[params[1]], kernel_name='rbf',
                      max_iter=10000, wss='wss3')
        return clf

    def eval(clf, train_x, train_y, valid_x, valid_y):
        clf.fit(train_x, train_y)
        score = clf.score(valid_x, valid_y)
        return score
    x = np.vstack((x_train, x_test))
    y = np.hstack((y_train, y_test))
    print y.shape
    intervals = [[0.5, 10], [0.01, 0.5]]
    opt = bo.BO(make=make, eval=eval, intervals=intervals, acqparams=1e-6,
                fold_num=2, opt_times=50)
    params, values = opt.fit(x, y)
    np.savetxt('bo_svm_params.txt', params)
    np.savetxt('bo_svm_values.txt', values)
    """

    clf = svc.SVC(C=1.0, kernel_name='rbf', wss='WSS1', params=[0.1])
    #clf = svm.SVC(C=1.0, kernel='rbf', gamma=0.1)
    s = time.time()
    clf.fit(x_train, y_train)
    e = time.time()
    print(e-s)
    print(clf.score(x_test, y_test))



