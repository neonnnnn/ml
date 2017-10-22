import torch
import torch.functional as F
from torch.autograd import Variable


def gaussian(mean, logvar):
    std = F.exp(0.5 * logvar)
    eps = Variable(std.data.new(std.size()).normal_())
    return mean + std*eps


def categorical(mean, temp):
    g = -torch.log(1e-10 - torch.log(1e-10+Variable(mean.data.new(mean.size()).uniform_())))
    if mean.ndim != 3:
        return F.softmax((torch.log(mean + 1e-10) + g)/temp)
    else:
        shape = (mean.size()[0] * mean.size()[1], mean.size(2))
        samples = F.softmax(((torch.log(mean + 1e-10) + g)/temp).view(shape))

        return samples.view_as(mean)


def bernoulli(mean, temp):
    g1 = -torch.log(1e-10 - torch.log(1e-10+Variable(mean.data.new(mean.shape).uniform_())))
    g2 = -torch.log(1e-10 - torch.log(1e-10+Variable(mean.data.new(mean.shape).uniform_())))
    return F.sigmoid((g1 + torch.log(1e-10+mean) - g2 - torch.log(1e-10+1-mean)) / temp)
