import theano.tensor as T


class CrossEntropy(object):
    def __init__(self, weight=1., reg=None):
        self.weight = weight
        self.reg = reg

    def get_output(self, y, p_y_given_x):
        loss = -T.mean(y * T.log(T.clip(p_y_given_x.ravel(), 1e-20, 1)) + (1 - y) * T.log(T.clip(1 - p_y_given_x.ravel(), 1e-20, 1)))
        return self.weight * loss


class MulticlassLogLoss(object):
    def __init__(self, weight=1.):
        self.weight = weight

    def get_output(self, y, p_y_given_x):
        # if categorical variables
        if y.ndim == 1:
            loss = -T.mean(T.log(1e-20 + p_y_given_x)[T.arange(y.shape[0]), y])
        # if one-hot
        elif y.ndim == 2:
            loss = -T.mean(T.sum(y * T.log(1e-20 + p_y_given_x), axis=1))
        # else
        else:
            raise Exception('Label Error:label must be scalar or vector. If not miss, you must rewrite model, objective etc.')

        return self.weight * loss


class KL(object):
    def __init__(self, weight=1.):
        self.weight = weight

    def get_output(self, p, q):
        loss = T.mean(p * T.log(p / q) + (1 - p) * T.log((1 - p) / (1 - q)))
        return self.weight * loss


class MeanSquaredError(object):
    def __init__(self, weight=1.):
        self.weight = weight

    def get_output(self, y, p_y_given_x):
        loss = T.mean(T.sum(T.square(y - p_y_given_x), axis=1))
        return self.weight * loss


class L2Regularization(object):
    def __init__(self, weight=0.0001):
        self.weight = weight

    def get_output(self, layers):
        L2_reg = 0
        for layer in layers:
            if hasattr(layer, 'W'):
                L2_reg += (layer.W ** 2).sum()

        return self.weight * L2_reg


class L1Regularization(object):
    def __init__(self, weight=0.0001):
        self.weight = weight

    def get_output(self, layers):
        L1_reg = 0
        for layer in layers:
            if hasattr(layer, 'W'):
                L1_reg += abs(layer.W).sum()

        return self.weight * L1_reg



