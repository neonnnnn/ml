import numpy as np
from ml.bo import bo
import load_mnist
from ml.deeplearning.layers import Dense, Activation
from ml.deeplearning.optimizers import SGD
from ml.deeplearning.objectives import MulticlassLogLoss
from ml.deeplearning.models import Sequential

if __name__ == '__main__':
    dataset = load_mnist.load_data()
    x_train, y_train = dataset[0]
    x_test, y_test = dataset[1]


    def make(params):
        rng = np.random.RandomState(1234)
        opt = SGD(lr=params[0], momentum=params[1])
        loss = MulticlassLogLoss()
        models = Sequential(784, loss=loss, rng=rng, opt=opt, batch_size=20, nb_epoch=100, iprint=False)
        models.add(Dense(10))
        models.add(Activation('softmax'))

        return models

    def eval(clf, train_x, train_y, valid_x, valid_y):
        clf.fit(train_x, train_y)
        score = clf.score(valid_x, valid_y)
        return score


    intervals = [[0.001, 1.], [0.1, 1.]]
    opt = bo.BO(make=make, eval=eval, intervals=intervals, opt_times=100, acq="UCB")
    params, values = opt.fit(x_train, y_train, x_test, y_test)
    # params, values = opt.fit(x_train, y_train, x_test, y_test)
    np.savetxt("bo_logistic_params.txt", params)
    np.savetxt("bo_logistic_values.txt", values)
