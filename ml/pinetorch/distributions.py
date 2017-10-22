from __future__ import absolute_import
import torch
import torch.functional as F
from abc import ABCMeta, abstractmethod
from torch import nn
from .sampler import gaussian, categorical, bernoulli
import math


class Distribution(nn.Module):
    __metaclass__ = ABCMeta

    @abstractmethod
    def log_likelihood(self, sample, *args, **kwargs):
        pass

    def log_likelihood_with_forward(self, sample, x, *args, **kwargs):
        params = self.forward(x, sampling=False, *args, **kwargs)
        return self.log_likelihood(sample, *params)

    def __call__(self, x, sampling=False, *args, **kwargs):
        return self.forward(x, sampling=sampling, *args, **kwargs)


class Gaussian(Distribution):
    def __init__(self, mean_layer, logvar_layer, network=None):
        super(Gaussian, self).__init__()
        self.network = network
        self.mean_layer = mean_layer
        self.logvar_layer = logvar_layer

    def forward(self, x, sampling=False):
        if self.network is not None:
            nn_output = self.network.forward(x)
        else:
            nn_output = x

        mean = self.mean_layer.forward(nn_output)
        logvar = self.logvar_layer.forward(nn_output)
        if not sampling:
            return mean, logvar
        else:
            z = gaussian(mean, logvar)
            return mean, logvar, z

    def log_likelihood(self, sample, mean, logvar):
        batch_size = sample.size()[0]
        c = - 0.5 * math.log(2 * math.pi)
        return torch.sum(c - logvar / 2 - (sample - mean) ** 2 / (2 * F.exp(logvar))) / batch_size


class Bernoulli(Distribution):
    def __init__(self, mean_layer, network=None, temp=1.0):
        super(Bernoulli, self).__init__()
        self.temp = temp
        self.mean_layer = mean_layer
        self.network = network

    def forward(self, x, sampling=True):
        if self.network is not None:
            nn_output = self.network.forward(x)
        else:
            nn_output = x

        mean = self.mean_layer.forward(nn_output)
        if not sampling:
            return mean
        else:
            z = bernoulli(mean, self.temp)
            return mean, z

    def log_likelihood(self, sample, mean):
        batch_size = sample.size()[0]
        return torch.sum(sample * torch.log(mean + 1e-10)
                         + (1 - sample) * torch.log(1 - mean + 1e-10)) / batch_size

    def log_likelihood_with_forward(self, sample, x, train=True, *args, **kwargs):
        mean = self.forward(x, sampling=False, *args, **kwargs)
        return self.log_likelihood(sample, mean)


class Categorical(Bernoulli):
    def forward(self, x, sampling=True):
        if self.network is not None:
            nn_output = self.network.forward(x)
        else:
            nn_output = x

        mean = F.softmax(self.mean_layer.forward(nn_output))
        if not sampling:
            return mean
        else:
            z = categorical(mean, temp=self.temp)
            return mean, z

    def log_likelihood(self, sample, mean):
        batch_size = sample.size()[0]
        return torch.sum(sample * torch.log(mean + 1e-10)) / batch_size
