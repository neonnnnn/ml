import theano.tensor as T


def gaussian(mean, std, srng):
    eps = srng.normal(mean.shape, dtype=mean.dtype)
    return mean + std*eps


def categorical(mean, temp, srng):
    g = -T.log(1e-10 - T.log(1e-10+srng.uniform(mean.shape)))
    if mean.ndim != 3:
        return T.nnet.softmax((T.log(mean + 1e-10) + g)/temp)
    else:
        true_shape = mean.shape
        samples = T.nnet.softmax(((T.log(mean + 1e-10) + g)/temp).reshape(
            true_shape[0]*true_shape[1], true_shape[2]
        ))

        return samples.reshape(mean.shape)


def bernoulli(mean, temp, srng):
    g1 = -T.log(1e-10 - T.log(1e-10+srng.uniform(mean.shape)))
    g2 = -T.log(1e-10 - T.log(1e-10+srng.uniform(mean.shape)))
    return T.nnet.sigmoid((g1 + T.log(1e-10+mean) - g2 - T.log(1e-10+1-mean))/temp)
