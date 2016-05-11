import numpy as np
import svc
import load_mnist
import time
from sklearn import svm


if __name__ == '__main__':
    dataset = load_mnist.load_data("../../../dataset/mnist.pkl.gz")
    x_train, y_train = dataset[0]
    x_test, y_test = dataset[1]

    x_train = x_train[np.where((y_train == 5) | (y_train == 8))]
    y_train = y_train[np.where((y_train == 5) | (y_train == 8))]
    y_train = ((y_train - 5) / 3) * 2 - 1

    x_test = x_test[np.where((y_test == 5) | (y_test == 8))]
    y_test = y_test[np.where((y_test == 5) | (y_test == 8))]
    y_test = ((y_test - 5) / 3) * 2 - 1
    # Set parameters

    max_iter = 10000
    C = 1.0
    # Set kernel function
    kernel = 'linear'

    wss = 'wss1'
    # Create object
    clf = svc.SVC(C=C, kernel_name=kernel, params=[0.005], max_iter=max_iter, wss=wss, eps=1e-3)
    #clf = svm.SVC(C=1, kernel='l', max_iter=max_iter, gamma=0.005, degree=2, coef0=1, shrinking=False)
    #clf = svm.LinearSVC(C=1, max_iter=max_iter)
    start = time.time()
    clf.fit(x_train, y_train)

    end = time.time()
    print end - start
    score = clf.score(x_test, y_test)
    pred = clf.predict(x_test)
    print ("score:"), (score)
