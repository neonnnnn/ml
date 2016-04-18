import numpy as np
import SVC
import load_mnist
import time
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
    # Set parameters
    max_iter = 50000
    C = 1e0
    # Set kernel function
    kernel = 'linear'
    shrinking = False
    # Create object
    clf = SVC.SVC(C=C, Kernel=kernel, params=[0.005], max_iter=max_iter)
    #clf = svm.SVC(C=1, kernel='linear', max_iter=max_iter, gamma=0.005, shrinking=shrinking)

    start = time.time()
    # Run SMO algorithm
    clf.fit(x_train, y_train)
    end = time.time()
    print end - start
    score = clf.score(x_test, y_test)
    print ("score:"), (score)
