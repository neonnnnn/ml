import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
import optimizers
from .. import utils
import scipy.sparse as sp
import objectives


class Sequential(object):
    def __init__(self, n_in, rng=np.random.RandomState(0), iprint=True):
        self.n_in = n_in
        self.rng = rng
        self.params = []
        self.layers = []
        self.loss = None
        self.opt = None
        self.batch_size = None
        self.nb_epoch = None
        self.train_function = None
        self.test_function = None
        self.iprint = iprint

    # add layer
    def add(self, this_layer, add_params=True):
        if isinstance(this_layer, Sequential):
            self.layers = self.layers + this_layer.layers
            if add_params:
                self.params = self.params + this_layer.params
        else:
            if hasattr(this_layer, 'rng'):
                this_layer.set_rng(self.rng)

            if len(self.layers) == 0:
                this_layer.set_input_shape(self.n_in)
            else:
                this_layer.set_input_shape(self.layers[len(self.layers)-1].n_out)

            if hasattr(this_layer, "params") and add_params:
                this_layer.set_params()
                self.params = self.params + this_layer.params

            self.layers = self.layers + [this_layer]

    # set output
    def get_top_output(self, x):
        output = self.layers[0].get_output(x)
        for layer in self.layers[1:]:
            output = layer.get_output(output)

        return output

    # set output for train
    def get_top_output_train(self, x):
        output = self.layers[0].get_output_train(x)
        for layer in self.layers[1:]:
            output = layer.get_output_train(output)

        return output

    def get_loss_output(self, y, output):
        if type(self.loss) == list:
            loss = 0.
            for l in self.loss:
                name = l.__class__.__name__
                if name == "L2Regularization" or name == "L1Regularization":
                    loss += l.get_output(self.layers)
                else:
                    loss += l.get_output(y, output)
        else:
            loss = self.loss.get_output(y, output)

        return loss

    # get train function
    def get_train_function(self, y_ndim):
        x = T.matrix('x')
        if self.layers[0].__class__.__name__ == 'Conv':
            x = x.reshape((self.batch_size, self.layers[0].n_in[0], self.layers[0].n_in[1], self.layers[0].n_in[2]))
        else:
            x = x.reshape((self.batch_size, self.layers[0].n_in))
        if y_ndim == 0:
            y = T.ivector('y')
        elif y_ndim == 1:
            y = T.matrix('y')
        output = self.get_top_output_train(x)
        cost = self.get_loss_output(y, output)
        updates = self.opt.get_update(cost, self.params)
        for layer in self.layers:
            if hasattr(layer, "updates"):
                updates += layer.updates
        return theano.function(inputs=[x, y], outputs=cost, updates=updates)

    # get pred function
    def get_test_function(self):
        x = T.matrix('x')
        if self.layers[0].__class__.__name__ == 'Conv':
            x = x.reshape((self.batch_size, self.layers[0].n_in[0], self.layers[0].n_in[1], self.layers[0].n_in[2]))
        else:
            x = x.reshape((self.batch_size, self.layers[0].n_in))
        output = self.get_top_output(x)
        return theano.function(inputs=[x], outputs=output)

    def _batch_train(self, x_train, y_train, train_model, n_batches):
        train_loss = []
        batch_start = 0
        batch_end = 0
        # if x is sparse matrix
        if sp.issparse(x_train):
            for i in xrange(n_batches):
                s = timeit.default_timer()
                batch_end = batch_start + self.batch_size
                train_loss += [train_model(x_train[batch_start:batch_end].toarray(), y_train[batch_start:batch_end])]
                batch_start += self.batch_size
                if self.iprint:
                    e = timeit.default_timer()
                    utils.progbar(i + 1, n_batches, e - s)
                    sys.stdout.write(" batches")
            if batch_end != x_train.shape[0]:
                train_loss += [train_model(x_train[-self.batch_size:].toarray(), y_train[-self.batch_size:])]
        else:
            for i in xrange(n_batches):
                s = timeit.default_timer()
                batch_end = batch_start + self.batch_size
                train_loss += [train_model(x_train[batch_start:batch_end], y_train[batch_start:batch_end])]
                batch_start += self.batch_size
                if self.iprint:
                    e = timeit.default_timer()
                    utils.progbar(i+1, n_batches, e - s)
                    sys.stdout.write(" batches")
            if batch_end != x_train.shape[0]:
                train_loss += [train_model(x_train[-self.batch_size:], y_train[-self.batch_size:])]
        if self.iprint:
            sys.stdout.write(', train_loss:%.5f' % np.mean(train_loss))

        return train_loss

    def _calc_error_rate(self, x, y, model, n_batches):
        error = 0.0
        batch_start = 0
        # if x is sparse matrix
        if sp.issparse(x):
            for i in xrange(n_batches):
                batch_end = batch_start + self.batch_size
                pred = model(x[batch_start:batch_end].toarray())
                error += utils.num_of_error(y[batch_start:batch_end], pred)
                batch_start += self.batch_size
            if batch_end != x.shape[0]:
                pred = model(x[-self.batch_size:].toarray())
                error += utils.num_of_error(y[batch_end:], pred[-x.shape[0] + batch_end:])
        else:
            for i in xrange(n_batches):
                batch_end = batch_start + self.batch_size
                pred = model(x[batch_start:batch_end])
                error += utils.num_of_error(y[batch_start:batch_end], pred)
                batch_start += self.batch_size
            if batch_end != x.shape[0]:
                pred = model(x[-self.batch_size:])
                error += utils.num_of_error(y[batch_end:], pred[-x.shape[0] + batch_end:])

        error_rate = 100. * error / (self.batch_size * n_batches)

        return error_rate

    def _calc_loss(self, x, y, model, n_batches):
        loss = []
        batch_start = 0
        # if x is sparse matrix
        if sp.issparse(x):
            for i in xrange(n_batches):
                batch_end = batch_start + self.batch_size
                pred = model(x[batch_start:batch_end].toarray())
                loss += [self.get_loss_output(y[batch_start:batch_end], pred)]
                batch_start += self.batch_size
            if batch_end != x.shape[0]:
                pred = model(x[-self.batch_size:].toarray())
                loss += [self.loss.get_output(y[batch_end:], pred[-x.shape[0] + batch_end:])]
        else:
            for i in xrange(n_batches):
                batch_end = batch_start + self.batch_size
                pred = model(x[batch_start:batch_end])
                loss += [self.get_loss_output(y[batch_start:batch_end], pred)]
                batch_start += self.batch_size
            if batch_end != x.shape[0]:
                pred = model(x[-self.batch_size:])
                loss += [self.get_loss_output(y[batch_end:], pred[-x.shape[0] + batch_end:])]

        return np.mean(loss)

    # define batch_size, nb_epoch, loss and optimization method
    def compile(self, batch_size=128, nb_epoch=100, opt=optimizers.SGD(), loss=objectives.MulticlassLogLoss()):
        if type(opt) == str:
            opt = optimizers.get_from_module(opt)

        self.opt = opt
        self.loss = loss
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.train_function = None
        self.test_function = None

        if self.iprint:
            print 'optimization:{0}'.format(self.opt.__class__.__name__)
            print 'batch_size:{0}'.format(self.batch_size)
            print 'nb_epoch:{0}'.format(self.nb_epoch)
            print 'n_layers:{0}'.format(len(self.layers))
            if isinstance(self.loss, list):
                str_loss = ''
                for l in loss:
                    str_loss += str(l.weigth) + l.__class__.__name__ + ' + '
                print 'loss:{0}'.format(str_loss[:-2])
            else:
                print 'loss:{0}'.format(loss.__class__.__name__)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, valid_mode='loss', shuffle=True):
        # get the output of each layers and define train_model
        if self.train_function is None:
            y_ndim = y_train[0].ndim
            self.train_function = self.get_train_function(y_ndim)

        train_model = self.train_function
        n_train_batches = x_train.shape[0] / self.batch_size
        valid_flag = False

        # if there are valid data, define valid_model and calc valid_loss
        if x_valid is not None and y_valid is not None:
            if self.test_function is None:
                self.test_function = self.get_test_function()
            valid_model = self.test_function
            best_valid_loss = np.inf
            n_valid_batches = x_valid.shape[0] / self.batch_size
            valid_flag = True
            if self.iprint:
                print ('validation:True')

        start_time = timeit.default_timer()
        # training start
        if self.iprint:
            print ('training ...')
        i = 0
        train_loss = []
        while i < self.nb_epoch:
            i += 1
            if shuffle:
                x_train, y_train = utils.shuffle(x_train, y_train)
            if self.iprint:
                print 'epoch:', i
            start_batch_time = timeit.default_timer()
            train_loss += [np.mean(self._batch_train(x_train, y_train, train_model, n_train_batches))]

            # if there are valid data, calc valid_error
            if valid_flag:
                if valid_mode == "error_rate":
                    this_valid_loss = self.calc_error_rate(x_valid, y_valid, valid_model, n_valid_batches)
                    if self.iprint:
                        sys.stdout.write(', valid_error_rate:{0:.4f}%%'.format(this_valid_loss))
                elif valid_mode == "loss":
                    this_valid_loss = self.calc_loss(x_valid, y_valid, valid_model, n_valid_batches)
                    if self.iprint:
                        sys.stdout.write(', valid_loss:{0:.4f}'.format(this_valid_loss))
                else:
                    raise Exception("valid_mode error: valid_mode must be error_rate or loss.")

                # if this_valid_loss is better than best_valid_loss
                if this_valid_loss < best_valid_loss:
                    best_valid_loss = this_valid_loss

            end_batch_time = timeit.default_timer()
            if self.iprint:
                sys.stdout.write(', {0:.2f}s'.format(end_batch_time - start_batch_time))
                sys.stdout.write("\n")

        # training end
        end_time = timeit.default_timer()
        if self.iprint:
            if valid_flag:
                print('Training complete. Best validation score of {0} %% '.format(best_valid_loss))
            else:
                print('Training complete.')
            print (' ran for {0:.2f}m'.format((end_time - start_time) / 60.))

        return train_loss

    def batch_fit(self, x_train, y_train):
        # get the output of each layers and define train_model
        if self.train_function is None:
            y_ndim = y_train[0].ndim
            self.train_function = self.get_train_function(y_ndim)

        train_model = self.train_function
        # if x is sparse matrix
        if sp.issparse(x_train):
            train_loss = train_model(x_train.toarray(), y_train)
        else:
            train_loss = train_model(x_train, y_train)

        return train_loss

    def predict(self, data_x):
        if self.test_function is None:
            self.test_function = self.get_test_function()

        test_model = self.test_function

        n_pred_batches = data_x.shape[0] / self.batch_size
        if isinstance(self.layers[-1].n_out, int):
            output = np.zeros((data_x.shape[0], self.layers[-1].n_out), dtype=theano.config.floatX)
        else:
            output_shape = [data_x.shape[0]] + list(self.layers[-1].n_out)
            output = np.zeros(output_shape, dtype=theano.config.floatX)

        batch_start = 0
        batch_end = 0
        # if data_x is sparse matrix
        if sp.issparse(data_x):
            for i in xrange(n_pred_batches):
                batch_end = batch_start + self.batch_size
                output[batch_start:batch_end] = test_model(data_x[batch_start:batch_end].toarray())
                batch_start += self.batch_size
            if batch_end != data_x.shape[0]:
                dummy_shape = data_x.shape
                dummy_shape[0] = data_x.shape[0] - batch_end
                dummy = np.zeros(dummy_shape, dtype=theano.config.floatX)
                pred = test_model(np.vstack(dummy, data_x[batch_end:].toarray()))
                output[batch_end:] = pred[-data_x.shape[0] + batch_end:]
        else:
            for i in xrange(n_pred_batches):
                batch_end += self.batch_size
                output[batch_start:batch_end] = test_model(data_x[batch_start:batch_end])
                batch_start += self.batch_size
            if batch_end != data_x.shape[0]:
                dummy_shape = data_x.shape
                dummy_shape[0] = data_x.shape[0] - batch_end
                dummy = np.zeros(dummy_shape, dtype=theano.config.floatX)
                pred = test_model(np.vstack(dummy, data_x[batch_end:]))
                output[batch_end:] = pred[-data_x.shape[0] + batch_end:]

        if self.layers[-1].n_out == 1:
            output = output.ravel()

        return output

    def accuracy(self, data_x, data_y):
        pred = self.predict(data_x)
        error = utils.num_of_error(data_y, pred)
        accuracy = 1 - (1.0 * error) / data_y.shape[0]
        return accuracy

    def save_weights(self, layer_id, filename):
        if hasattr(self.layers[layer_id], 'W'):
            np.save(filename+'_W', self.layers[layer_id].W.get_value())
        if hasattr(self.layers[layer_id], 'b'):
            np.save(filename+'_b', self.layers[layer_id].b.get_value())
        else:
            print ('layer{0} doesnt have weights.'.format(layer_id))
