import numpy as np
from ml.bo import bo
import load_mnist
from ml.deeplearning.layers import Dense, Activation
from ml.deeplearning.optimizers import SGD
from ml.deeplearning.objectives import MulticlassLogLoss, L2Regularization
from ml.deeplearning.models import Sequential

if __name__ == '__main__':
    dataset = load_mnist.load_data()
    x_train, y_train = dataset[0]
    x_test, y_test = dataset[1]


    def make(params):
        rng = np.random.RandomState(1234)
        opt = SGD(lr=params[0], momentum=params[1])
        loss = [MulticlassLogLoss(), L2Regularization(params[2])]
        clf = Sequential(784, loss=loss, rng=rng, opt=opt, batch_size=20, nb_epoch=100, iprint=False)
        clf.add(Dense(10))
        clf.add(Activation('softmax'))

        return clf

    def eval(clf, train_x, train_y, valid_x, valid_y):
        clf.fit(train_x, train_y)
        score = clf.score(valid_x, valid_y)
        return score

    intervals = [[0.001, 1.], [0.01, 1.], [0.001, 0.01]]
    grid = [1000, 100, 10]

    opt = bo.BO(make=make, eval=eval, intervals=intervals, grid=grid, opt_times=100, acq="EI")
    params, values = opt.fit(x_train, y_train, x_test, y_test)
    # params, values = opt.fit(x_train, y_train, x_test, y_test)
    np.savetxt("bo_logistic_params_EI.txt", params)
    np.savetxt("bo_logistic_values_EI.txt", values)
    del opt

    opt = bo.BO(make=make, eval=eval, intervals=intervals, grid=grid, opt_times=100, acq="UCB")
    params, values = opt.fit(x_train, y_train, x_test, y_test)
    # params, values = opt.fit(x_train, y_train, x_test, y_test)
    np.savetxt("bo_logistic_params_UCB.txt", params)
    np.savetxt("bo_logistic_values_UCB.txt", values)
    del opt

    opt = bo.BO(make=make, eval=eval, intervals=intervals, grid=grid, opt_times=100, acq="MI")
    params, values = opt.fit(x_train, y_train, x_test, y_test)
    # params, values = opt.fit(x_train, y_train, x_test, y_test)
    np.savetxt("bo_logistic_params_MI.txt", params)
    np.savetxt("bo_logistic_values_MI.txt", values)
