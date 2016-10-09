import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
import optimizers
from ..utils import BatchIterator, progbar, num_of_error
import scipy.sparse as sp
import objectives
import inspect
from abc import ABCMeta, abstractmethod


# filter out layers which dont have method
def filter_method(method, origin):
    return filter(lambda x: hasattr(x, method), origin)


def map_method(method, origin, *args):
    filteredlist = filter_method(method, origin)
    if len(args) == 0:
        ret = map(lambda x: getattr(x, method)(), filteredlist)
    elif len(args) == 1:
        ret = map(lambda x: getattr(x, method)(args[0]), filteredlist)
    else:
        ret = map(lambda x, y: getattr(x, method)(y), filteredlist, args)
    return ret


def run(inputs, function, iprint=True):
    output = []
    if iprint:
        n_batches = inputs.n_batches
        s = timeit.default_timer()
        i = 0
    for batch in inputs:
        output += function(*[b if not sp.issparse(b) else b.toarray() for b in batch])
        if iprint:
            e = timeit.default_timer()
            progbar(i+1, n_batches, e - s)
            i += 1
    return output


def onebatch_run(inputs, function):
    output = function(*[b if not sp.issparse(b) else b.toarray() for b in [inputs]])
    return output


class Sequential(object):
    def __init__(self, n_in, rng=np.random.RandomState(0), iprint=True):
        self.n_in = n_in
        self.n_out = None
        self.rng = rng
        self.params = []
        self.layers = []
        self.loss = None
        self.opt = None
        self.batch_size = None
        self.nb_epoch = None
        self.train_function = None
        self.pred_function = None
        self.test_function = None
        self.iprint = iprint

    def __call__(self, x, train=True):
        self.forward(x, train)

    # add layer
    def add(self, this_layer, add_params=True):
        if isinstance(this_layer, Sequential):
            if add_params:
                self.params += this_layer.params
            self.layers = self.layers + this_layer.layers
        else:
            # set rng
            if hasattr(this_layer, 'set_rng'):
                this_layer.set_rng(self.rng)
            # set input shape
            if len(self.layers) == 0:
                this_layer.set_input_shape(self.n_in)
            else:
                this_layer.set_input_shape(self.layers[len(self.layers) - 1].n_out)
            # set params
            if hasattr(this_layer, "params"):
                if this_layer.params is None:
                    this_layer.set_params()
                if add_params:
                    self.params += this_layer.params
            self.layers = self.layers + [this_layer]

    # set output
    def forward(self, x, train=True):
        return reduce(lambda a, b: b.forward(a, train), [x] + self.layers)

    def function(self, mode='train', y_ndim=None):
        x = T.matrix('x')
        if isinstance(self.n_in, int):
            x = x.reshape((self.batch_size, self.n_in))
        else:
            x = x.reshape([self.batch_size] + list(self.n_in))

        if mode == 'train' or mode == 'test':
            if y_ndim is None:
                raise ValueError('If mode is "train" or "test", you set y_ndim')
            if y_ndim == 0:
                y = T.ivector('y')
            else:
                y = T.matrix('y')

            if mode == 'train':
                output = self.forward(x, train=True)
                cost = self.get_loss_output(y, output)
                updates = self.opt.get_update(cost, self.params)
                for layer in self.layers:
                    if hasattr(layer, "updates"):
                        updates += layer.updates
                function = theano.function(inputs=[x, y], outputs=[cost], updates=updates)
            else:
                output = self.forward(x, train=False)
                cost = self.get_loss_output(y, output)
                function = theano.function(inputs=[x, y], outputs=[cost])
        elif mode == 'pred':
            output = self.forward(x, train=False)
            function = theano.function(inputs=[x], outputs=[output])
        else:
            raise ValueError('mode must be "train" or "test" or "pred".')

        return function

    def get_loss_output(self, y, output):
        if type(self.loss) == list:
            loss = 0.
            for l in self.loss:
                args = inspect.getargspec(l.calc)[0]
                if len(args) == 1:
                    loss += l.calc()
                elif len(args) == 2:
                    loss += l.calc(self.layers)
                else:
                    loss += l.calc(y, output)
        else:
            args = inspect.getargspec(self.loss.calc)[0]
            if len(args) == 1:
                loss = self.loss.calc()
            elif len(args) == 2:
                loss = self.loss.calc(self.layers)
            else:
                loss = self.loss.calc(y, output)

        return loss

    # define batch_size, nb_epoch, loss and optimization method
    def compile(self, batch_size=128, nb_epoch=100, opt=optimizers.SGD(), loss=objectives.MulticlassLogLoss()):
        self.opt = opt
        self.loss = loss
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.train_function = None
        self.pred_function = None

        if self.iprint:
            print 'optimization:{0}'.format(self.opt.__class__.__name__)
            print 'batch_size:{0}'.format(self.batch_size)
            print 'nb_epoch:{0}'.format(self.nb_epoch)
            print 'n_layers:{0}'.format(len(self.layers))
            if isinstance(self.loss, list):
                str_loss = ''
                for l in self.loss:
                    str_loss += str(l.weight) + l.__class__.__name__ + ' + '
                print 'loss:{0}'.format(str_loss[:-2])
            else:
                print 'loss:{0}'.format(loss.__class__.__name__)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, valid_mode='loss', shuffle=True):
        # get the output of each layers and define train_model
        if self.train_function is None:
            y_ndim = y_train[0].ndim
            self.train_function = self.function('train', y_ndim)

        train_iter = BatchIterator((x_train, y_train), batch_size=self.batch_size, shuffle=shuffle)
        valid_flag = False

        # if there are valid data, define valid_model and calc valid_loss
        if x_valid is not None and y_valid is not None:
            if valid_mode == 'error_rate' and self.pred_function is None:
                self.pred_function = self.function('pred')
                valid_iter = BatchIterator(x_valid, batch_size=self.batch_size, shuffle=False)
            elif valid_mode == 'loss' and self.test_function is None:
                self.test_function = self.function('test', y_valid[0].ndim)
                valid_iter = BatchIterator((x_valid, y_valid), batch_size=self.batch_size, shuffle=False)
            else:
                raise Exception('valid_mode error: valid_mode must be "error_rate" or "loss".')
            best_valid_loss = np.inf
            valid_flag = True

        start_time = timeit.default_timer()
        # training start
        if self.iprint:
            print ('training ...')
        i = 0
        train_loss = []

        # training while i < nb_epoch
        while i < self.nb_epoch:
            i += 1
            if self.iprint:
                print 'epoch:', i
            train_loss += [np.mean(run(train_iter, self.train_function, iprint=True))]
            sys.stdout.write(', train_loss:{0:.5f}'.format(train_loss[-1]))
            # if there are valid data, calc valid_error
            if valid_flag:
                if valid_mode == 'error_rate':
                    pred = self.predict(valid_iter)
                    this_valid_loss = (1.0 * num_of_error(y_valid, pred)) / y_valid.shape[0]
                elif valid_mode == "loss":
                    this_valid_loss = self.__test(valid_iter, self.test_function)
                if self.iprint:
                    sys.stdout.write(', valid_{0}:{1:.5f}'.format(valid_mode, this_valid_loss))

                # if this_valid_loss is better than best_valid_loss
                if this_valid_loss < best_valid_loss:
                    best_valid_loss = this_valid_loss

            if self.iprint:
                sys.stdout.write("\n")

        # training end
        end_time = timeit.default_timer()
        if self.iprint:
            if valid_flag:
                print('Training complete. Best validation score of {0}'.format(best_valid_loss))
            else:
                print('Training complete.')
            print (' ran for {0:.2f}m'.format((end_time - start_time) / 60.))

        return train_loss

    def onebatch_fit(self, x_train, y_train):
        # get the output of each layers and define train_model
        if self.train_function is None:
            y_ndim = y_train[0].ndim
            self.train_function = self.function('train', y_ndim)

        train_model = self.train_function
        # if x is sparse matrix
        if sp.issparse(x_train):
            train_loss = train_model(x_train.toarray(), y_train)
        else:
            train_loss = train_model(x_train, y_train)

        return train_loss

    def predict(self, data_x):
        if self.pred_function is None:
            self.pred_function = self.function('pred')
        if isinstance(data_x, BatchIterator):
            data_iter = data_x
        else:
            data_iter = BatchIterator(data_x, self.batch_size, False)
        pred = self.__predict(data_iter, self.pred_function)

        return pred

    def __predict(self, data_iter, function):
        n_samples = data_iter.n_samples
        output = run(data_iter, function, False)
        if len(output) == 1:
            pred = output[0]
        else:
            pred = reduce(lambda x, y: np.vstack((x, y)), output[:-1])
            if pred.shape[0] + output[-1].shape[0] == n_samples:
                pred = np.vstack((pred, output[-1]))
            else:
                pred = np.vstack((pred, output[-1][-(n_samples - pred.shape[0]):]))

        if self.layers[-1].n_out == 1:
            pred = pred.ravel()

        return pred

    def test(self, data_x, data_y, mode='mean'):
        if self.test_function is None:
            self.test_function = self.function('test', data_y[0].ndim)
        test_iter = BatchIterator((data_x, data_y), self.batch_size, False)
        output = self.__test(test_iter, self.test_function, mode)

        return np.mean(output)

    def __test(self, data_iter, function, mode='mean'):
        output = run(data_iter, function, False)
        if mode == 'mean' or mode:
            output = np.mean(output)
        elif mode == 'sum' or mode:
            output = np.sum(output)
        else:
            raise ValueError('mode must be "mean" or 1, or "sum" or 0.')

        return output

    def accuracy(self, data_x, data_y):
        pred = self.predict(data_x)
        error = num_of_error(data_y, pred)
        accuracy = 1 - (1.0 * error) / data_y.shape[0]
        return accuracy

    def save_weights(self, layer_id, filename):
        if hasattr(self.layers[layer_id], 'W'):
            np.save(filename+'_W', self.layers[layer_id].W.get_value())
        if hasattr(self.layers[layer_id], 'b'):
            np.save(filename+'_b', self.layers[layer_id].b.get_value())
        else:
            print ('layer{0} doesnt have weights.'.format(layer_id))


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self, rng, **kwargs):
        self.params = []
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.set_input_shape()
        self.set_rng(rng)
        self.set_params()

    def set_params(self):
        paramslayers = filter(lambda x: hasattr(x, 'params'), map(lambda x: getattr(self, x), self.__dict__.keys()))
        map(lambda x: x.set_params(), filter(lambda x: hasattr(x, 'set_params'), paramslayers))
        self.params = reduce(lambda x, y: x + y, map(lambda x: x.params, paramslayers))

    def set_input_shape(self):
        layers = filter(lambda x: hasattr(x, 'set_input_shape'), map(lambda x: getattr(self, x), self.__dict__.keys()))
        map(lambda x: x.set_input_shape(x.n_in), layers)

    def set_rng(self, rng):
        rnglayers = filter(lambda x: hasattr(x, 'set_rng'), map(lambda x: getattr(self, x), self.__dict__.keys()))
        map(lambda x: x.set_rng(rng), rnglayers)

    def updates(self, cost, opt):
        updates = opt.get_update(cost, self.params)
        updatelayers = filter(lambda x: hasattr(x, 'updates'), map(lambda x: getattr(self, x), self.__dict__.keys()))
        for layer in updatelayers:
            updates += layer.updates
        return updates

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        pass

    @abstractmethod
    def function(self, *inputs, **kwargs):
        pass
