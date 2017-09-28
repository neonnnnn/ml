import sys
import timeit
import numpy as np
import theano
from optimizers import Optimizer
from layers import Layer, Concat
from theanoutils import variable, run, run_on_batch
from ..utils import num_of_error
import inspect
from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self, rng, **layers):
        self.params = []
        # self.updates is list for params which have been updated
        # not by gradient descent(e.g., mean_inf and var_inf in BatchNorm).
        self.updates = []
        self.rng = rng
        for key, value in layers.items():
            setattr(self, key, value)

        self.opt = None
        self.train_loss = None
        self.test_loss = None
        self.train_function = None
        self.pred_function = None
        self.test_function = None

        # Because the instance of Model class is set shape, rng and parameters
        # in __init__, Model class doesn't have set_shape, set_rng and
        # set_params methods.
        self._set_shape()
        self._set_rng()
        self._set_params()

    def __call__(self, x, train=True):
        return self.forward(x, train)

    def get_config(self):
        config = {'class': self.__class__.__name__}
        return config

    def get_layers_with_names_configs(self):
        paramslayers = filter(lambda v: hasattr(v, 'params'), self.__dict__.values())
        names = filter(lambda k: hasattr(self.__dict__[k], 'params'), self.__dict__.keys())
        config = OrderedDict()
        for layer, name in zip(paramslayers, names):
            config[name] = layer.get_config()
        return paramslayers, names, config

    def _set_params(self):
        paramslayers = filter(lambda x: hasattr(x, 'params'),
                              self.__dict__.values())
        map(lambda x: x.set_params(),
            filter(lambda x: hasattr(x, 'set_params'), paramslayers))
        self.params = reduce(lambda x, y: x + y,
                             map(lambda x: x.params, paramslayers))

    def _set_shape(self):
        layers = filter(lambda x: hasattr(x, 'set_shape'), self.__dict__.values())
        map(lambda x: x.set_shape(x.n_in), layers)

    def _set_rng(self):
        rnglayers = filter(lambda x: hasattr(x, 'set_rng'),
                           self.__dict__.values())
        map(lambda x: x.set_rng(self.rng), rnglayers)

    def get_updates(self):
        updatelayers = filter(
            lambda x: (hasattr(x, 'get_updates') and (not isinstance(x, Optimizer))),
            self.__dict__.values()
        )
        for layer in updatelayers:
            self.updates += layer.get_updates()
        return self.updates

    def compile(self, opt, train_loss, test_loss=None):
        self.opt = opt
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.train_function = None
        self.pred_function = None
        self.test_function = None

    def fit(self, train_batches, epoch=100, valid_batches=None, iprint=True):
        variables = [variable(datum) for datum in train_batches.data]

        if self.train_function is None:
            self.train_function = self.function(*variables, mode='train')
        if self.test_function is None and valid_batches is not None:
            self.test_function = self.function(*variables, mode='test')

        if iprint:
            header = 'epoch' + ' '*3 + 'train_loss'
            if valid_batches is not None:
                header += ' '*3 + 'valid_loss'
            header += ' '*3 + 'time'
            print(header)

        return self._fit(train_batches, epoch, valid_batches, iprint)

    def fit_on_batch(self, train_batch):
        variables = [variable(datum) for datum in train_batch]
        if self.train_function is None:
            self.train_function = self.function(*variables, mode='train')
        return run_on_batch(train_batch, self.train_function)

    def _fit(self, train_batches, epoch, valid_batches=None, iprint=True):
        train_loss = []
        if valid_batches is not None:
            valid_loss = []
        if iprint:
            start_time = timeit.default_timer()
        # training while i < epoch
        for i in xrange(epoch):
            start_epoch = timeit.default_timer()
            loss = run(train_batches, self.train_function, iprint)
            train_loss += [np.mean(loss)]
            if iprint:
                n_space = 8 - len(str(i+1))
                sys.stdout.write('{0}{1}'.format(i+1, ' ' * n_space))
                n_space = 13 - len('{0:.5f}'.format(train_loss[-1]))
                sys.stdout.write('{0:.5f}{1}'.format(train_loss[-1],
                                                     ' ' * n_space))
            # if there are valid data, calc valid_error
            if valid_batches is not None:
                valid_loss += [self.__test(valid_batches)]
                # if this_valid_loss is better than best_valid_loss
                if iprint:
                    n_space = 13 - len('{0:.5f}'.format(valid_loss[-1]))
                    sys.stdout.write('{0:.5f}{1}'.format(valid_loss[-1],
                                                         ' ' * n_space))
            if iprint:
                end_epoch = timeit.default_timer()
                sys.stdout.write('{0:.2f}s\n'.format(end_epoch - start_epoch))

        # training end
        end_time = timeit.default_timer()
        if iprint:
            if valid_batches is not None:
                print('Best validation score: {0:.5f}'.format(min(valid_loss)))
            print(' ran for {0:.2f}m'.format((end_time - start_time)/60.))

        if valid_batches is not None:
            return train_loss, valid_loss
        else:
            return train_loss

    def predict(self, pred_batches):
        if self.pred_function is None:
            variables = [variable(datum) for datum in pred_batches.data]
            self.pred_function = self.function(*variables, mode='pred')

        return self.__predict(pred_batches)

    def __predict(self, data_batches):
        n_samples = data_batches.n_samples
        output = run(data_batches, self.pred_function, False)
        if len(output) == 1:
            pred = output[0]
        else:
            pred = reduce(lambda x, y: np.vstack((x, y)), output[:-1])
            if pred.shape[0] + output[-1].shape[0] == n_samples:
                pred = np.vstack((pred, output[-1]))
            else:
                pred = np.vstack((pred, output[-1][-(n_samples - pred.shape[0]):]))

        return pred

    def test(self, test_batches):
        if self.test_function is None:
            variables = [variable(datum) for datum in test_batches.data]
            self.test_function = self.function(*variables, mode='test')
        return self.__test(test_batches)

    def __test(self, test_batches):
        output = run(test_batches, self.test_function, False)
        output = np.mean(output)
        return output

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def function(self, *args, **kwargs):
        pass


class Sequential(Model):
    def __init__(self, n_in, rng=np.random.RandomState(0)):
        self.n_in = n_in
        self.n_out = n_in
        self.rng = rng
        self.params = []
        self.layers = []
        self.layer_names = []
        self.train_loss = None
        self.test_loss = None
        self.opt = None
        self.train_function = None
        self.pred_function = None
        self.test_function = None
        self.updates = []
        self.id_dict = defaultdict(int)

    # add layer
    def add(self, this_layer, add_params=True):
        if isinstance(this_layer, Sequential):
            if add_params:
                self.params += this_layer.params
            self.layers = self.layers + this_layer.layers
        elif isinstance(this_layer, Layer):
            # set rng
            if hasattr(this_layer, 'set_rng'):
                this_layer.set_rng(self.rng)
            # set input shape
            if len(self.layers) == 0:
                this_layer.set_shape(self.n_in)
            else:
                this_layer.set_shape(self.layers[len(self.layers) - 1].n_out)
            # set params
            if hasattr(this_layer, 'params'):
                if this_layer.params is None:
                    this_layer.set_params()
                if add_params:
                    self.params += this_layer.params
            self.layers = self.layers + [this_layer]
        else:
            TypeError('The added instance must be an instance of Layer or '
                      'Sequential class, Not' + type(this_layer) + '.')

        self.n_out = self.layers[-1].n_out

    def concat(self, layers, add_params=True):
        if len(self.layers) != 0:
            raise Exception('concat must do before add.')

        concat = Concat(layers)

        if isinstance(add_params, bool):
            add_params = [add_params] * len(layers)

        for layer, flag, n_in in zip(layers, add_params, self.n_in):
            if hasattr(layer, 'set_rng'):
                layer.set_rng(self.rng)
            if hasattr(layer, 'set_shape'):
                layer.set_shape(n_in)
            if hasattr(layer, 'params'):
                if layer.params is None:
                    layer.set_params()
                if add_params:
                    self.params += layer.params

        self.add(concat)

    # set output
    def forward(self, x, train=True):
        if hasattr(x, 'len'):
            if len(x) == 1:
                x = x[0]
        output = reduce(lambda a, b: b.forward(a, train=train), [x] + self.layers)

        return output

    # get update list (not updated by optimizer. e.g., mean_inf in BatchNorm)
    def get_updates(self):
        if len(self.updates) == 0:
            for layer in self.layers:
                if hasattr(layer, 'get_updates'):
                    self.updates += layer.get_updates()
        return self.updates

    def _make_train_function(self, inputs):
        x = inputs[:-1]
        y = inputs[-1]
        output = self.forward(x, train=True)
        loss = self._get_loss_output(y, output, self.train_loss)
        updates = self.opt.get_updates(loss, self.params)
        updates += self.get_updates()
        function = theano.function(inputs=list(inputs), outputs=[loss], updates=updates)
        return function

    def _make_test_function(self, inputs):
        x = inputs[:-1]
        y = inputs[-1]
        output = self.forward(x, train=False)
        if self.test_loss is None:
            test_loss = self.train_loss
        else:
            test_loss = self.test_loss
        loss = self._get_loss_output(y, output, test_loss)
        function = theano.function(inputs=list(inputs), outputs=[loss])
        return function

    def _make_pred_function(self, inputs):
        output = self.forward(inputs, train=False)
        function = theano.function(inputs=list(inputs), outputs=[output])
        return function

    def function(self, *inputs, **kwargs):
        if not ('mode' in kwargs):
            raise ValueError('When making a function, keyword args "mode" must be specified.')
        mode = kwargs['mode']
        if not (kwargs['mode'] in ['train', 'test', 'pred']):
            raise ValueError('mode must be "train" or "test" or "pred".')
        else:
            func = getattr(self, '_make_' + mode + '_function')(inputs)
        return func

    def _get_loss_output(self, y, output, losses):
        dic = {'y': y, 'output': output, 'layers': self.layers}
        if isinstance(losses, (tuple, list)):
            loss = 0.
            for l in losses:
                keys = inspect.getargspec(l.calc)[0]
                loss += l.calc(*[dic[key] for key in keys[1:]])
        else:
            keys = inspect.getargspec(losses.calc)[0]
            loss = losses.calc(*[dic[key] for key in keys[1:]])

        return loss

    def accuracy(self, x_batches, y):
        pred = self.predict(x_batches)
        error = num_of_error(y, pred)
        accuracy = 1 - (1.0*error)/y.shape[0]
        return accuracy

    def get_config(self):
        config = {'class': self.__class__.__name__, 'n_in': self.n_in}

        return config

    def get_layers_with_names_configs(self):
        names = []
        self.id_dict = defaultdict(int)
        for layer in self.layers:
            name = layer.__class__.__name__
            self.id_dict[name] += 1
            names.append(name + '_' + str(self.id_dict[name]))
        config = OrderedDict()
        for layer, name in zip(self.layers, names):
            config[name] = layer.get_config()
        return self.layers, names, config
