import torch
import torch.functional as F
from torch.autograd import Variable


def gaussian(mean, logvar):
    std = F.exp(logvar.mul(0.5))
    eps = Variable(std.data.new(std.size()).normal_())
    return mean + std*eps


def categorical(mean, temp):
    g = -torch.log(1e-10 - torch.log(1e-10+Variable(mean.data.new(mean.shape).uniform_())))
    if mean.ndim != 3:
        return F.softmax((torch.log(mean + 1e-10) + g)/temp)
    else:
        true_shape = mean.size()
        samples = F.softmax(((torch.log(mean + 1e-10) + g)/temp).view(
            true_shape[0]*true_shape[1], true_shape[2]
        ))

        return samples.view_as(mean)


def bernoulli(mean, temp):
    g1 = -torch.log(1e-10 - torch.log(1e-10+Variable(mean.data.new(mean.shape).uniform_())))
    g2 = -torch.log(1e-10 - torch.log(1e-10+Variable(mean.data.new(mean.shape).uniform_())))
    return F.sigmoid((g1 + torch.log(1e-10+mean) - g2 - torch.log(1e-10+1-mean)).div_(temp))
