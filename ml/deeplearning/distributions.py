from __future__ import absolute_import
import theano
import theano.tensor as T
from .models import Model
from abc import ABCMeta, abstractmethod
from .activations import softmax, sigmoid
from .sampler import gaussian, categorical, bernoulli
import math
from .theanoutils import sharedasarray, sharedzeros, variable


class Distribution(Model):
    __metaclass__ = ABCMeta

    def __init__(self, rng, **kwargs):
        self.sampling_function = None
        super(Distribution, self).__init__(rng, **kwargs)

    @abstractmethod
    def log_likelihood(self, sample, *args, **kwargs):
        pass

    def log_likelihood_with_forward(self, sample, x, train=True, *args, **kwargs):
        params = self.forward(x, train=train, sampling=False, *args, **kwargs)
        return self.log_likelihood(sample, *params)

    def __call__(self, x, sampling=False, train=True, *args, **kwargs):
        return self.forward(x, sampling=sampling, train=train, *args, **kwargs)


class Gaussian(Distribution):
    def __init__(self, mean_layer, logvar_layer, network=None, rng=None):

        if rng is None:
            rng = network.rng
        self.srng = T.shared_randomstreams.RandomStreams(rng.randint(99999))
        mean_layer.set_shape(network.n_out)
        logvar_layer.set_shape(network.n_out)
        self.n_out = mean_layer.n_out
        self.sampling_function = None
        super(Gaussian, self).__init__(rng,
                                       mean_layer=mean_layer,
                                       logvar_layer=logvar_layer,
                                       network=network)

    def forward(self, x, sampling=False, train=True, *args, **kwargs):
        if self.network is not None:
            nn_output = self.network.forward(x, train=train, *args, **kwargs)
        else:
            nn_output = x

        mean = self.mean_layer.forward(nn_output, train=train)
        logvar = self.logvar_layer.forward(nn_output, train=train)
        if not sampling:
            return mean, logvar
        else:
            z = gaussian(mean, T.exp(0.5*logvar), self.srng)
            return mean, logvar, z

    def predict(self, pred_batches, mode='pred', *args, **kwargs):
        if not (mode in ['pred', 'mean', 'logvar', 'sampling']):
            raise ValueError('mode must be "pred" (= "mean"), "logvar" or "sampling".')

        if self.pred_function is None or self.pred_function.name != mode:
            variables = [variable(datum) for datum in pred_batches.data]
            self.pred_function = self.function(*variables, mode=mode)

        return super(Gaussian, self).predict(pred_batches)

    def function(self, x, y=None, mode='train', *args, **kwargs):
        if mode == 'train':
            mean, logvar = self.forward(x, sampling=False, train=True)
            log_likelihood = T.mean(self.log_likelihood(y, mean, logvar))
            updates = self.opt.get_updates(-log_likelihood, self.params)
            updates += self.get_updates()
            return theano.function(inputs=[x], outputs=[-log_likelihood],
                                   updates=updates, name=mode)
        elif mode == 'pred' or 'mean':
            mean, _ = self.forward(x, sampling=False, train=False)
            return theano.function(inputs=[x], outputs=[mean], name=mode)
        elif mode == 'logvar':
            _, logvar = self.forward(x, sampling=False, train=False)
            return theano.function(inputs=[x], outputs=[logvar], name=mode)
        elif mode == 'test':
            mean, logvar = self.forward(x, sampling=False, train=False)
            log_likelihood = T.mean(self.log_likelihood(y, mean, logvar))
            return theano.function(inputs=[x], outputs=[-log_likelihood], name=mode)
        elif mode == 'sampling':
            mean, logvar, z = self.forward(x, sampling=True, train=False)
            return theano.function(inputs=[x], outputs=[z], name=mode)
        else:
            raise ValueError('mode must be "train", "test", "pred" (= "mean"),'
                             ' "logvar" or "sampling.')

    def log_likelihood(self, sample, mean, logvar):
        axis = tuple(range(sample.ndim))[1:]
        c = - 0.5 * math.log(2 * math.pi)
        return T.sum(c - logvar / 2 - (sample - mean) ** 2 / (2 * T.exp(logvar)), axis=axis)

    def log_likelihood_with_forward(self, sample, x, train=True, *args, **kwargs):
        mean, logvar = self.forward(x, train=train, sampling=False, *args, **kwargs)
        return self.log_likelihood(sample, mean, logvar)


class Bernoulli(Distribution):
    def __init__(self, mean_layer, network, temp=1., rng=None, annealing=None):
        if rng is None:
            rng = network.rng
        self.temp = sharedasarray(temp)
        self.srng = T.shared_randomstreams.RandomStreams(rng.randint(9999))
        mean_layer.set_shape(network.n_out)
        self.n_out = mean_layer.n_out
        self.annealing = annealing
        self.i = None
        super(Bernoulli, self).__init__(rng, mean_layer=mean_layer, network=network)

    def get_updates(self):
        super(Distribution, self).get_updates()
        if self.annealing is not None:
            if self.i is None:
                self.i = sharedasarray(0)
                self.updates.append((self.i, self.i + 1))
            self.updates.append((self.temp, self.annealing(self.temp, self.i)))

        return self.updates

    def forward(self, x, sampling=False, train=True, *args, **kwargs):
        if self.network is not None:
            nn_output = self.network.forward(x, train=train, *args, **kwargs)
        else:
            nn_output = x
        mean = sigmoid(self.mean_layer.forward(nn_output, train=train))
        if not sampling:
            return mean
        else:
            z = bernoulli(mean, temp=self.temp, srng=self.srng)
            return mean, z

    def function(self, x, y=None, mode='train', *args, **kwargs):
        if mode == 'train':
            mean = self.forward(x, sampling=False, train=True)
            log_likelihood = T.mean(self.log_likelihood(y, mean))
            updates = self.opt.get_updates(-log_likelihood, self.params)
            updates += self.get_updates()
            return theano.function(inputs=[x], outputs=[-log_likelihood], updates=updates)
        elif mode == 'pred':
            mean = self.forward(x, sampling=False, train=False)
            return theano.function(inputs=[x], outputs=[mean])
        elif mode == 'test':
            mean = self.forward(x, sampling=False, train=False)
            log_likelihood = T.mean(self.log_likelihood(y, mean))
            return theano.function(inputs=[x], outputs=[-log_likelihood])
        elif mode == 'sampling':
            _, z = self.forward(x, sampling=True, train=False)
            return theano.function(inputs=[x], outputs=[z])
        else:
            raise ValueError('mode must be "train" or "test" or "pred".')

    def log_likelihood(self, sample, mean):
        axis = tuple(range(sample.ndim))[1:]
        return T.sum(sample * T.log(mean + 1e-10) + (1 - sample) * T.log(1 - mean + 1e-10),
                     axis=axis)

    def log_likelihood_with_forward(self, sample, x, train=True, *args, **kwargs):
        mean = self.forward(x, train=train, sampling=False, *args, **kwargs)
        return self.log_likelihood(sample, mean)


class Categorical(Bernoulli):
    def forward(self, x, sampling=False, train=True, *args, **kwargs):
        if self.network is not None:
            nn_output = self.network.forward(x, train=train, *args, **kwargs)
        else:
            nn_output = x
        mean = softmax(self.mean_layer.forward(nn_output, train=train))
        if not sampling:
            return mean
        else:
            z = categorical(mean, temp=self.temp, srng=self.srng)
            return mean, z

    def log_likelihood(self, sample, mean):
        axis = tuple(range(sample.ndim))[1:]
        return T.sum(sample*T.log(mean + 1e-10), axis=axis)
