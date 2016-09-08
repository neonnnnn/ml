import theano.tensor as T


class CrossEntropy(object):
    def __init__(self, weight=1., mode=0):
        self.weight = weight
        self.mode = mode
        if not (self.mode == 0 or self.mode == 1):
            raise ValueError("mode must be 0 or 1.")

    def get_output(self, y, output):
        if y.ndim == 1:
            loss = -(y * T.log(T.clip(output.ravel(), 1e-20, 1)) + (1 - y) * T.log(T.clip(1 - output.ravel(), 1e-20, 1)))
        else:
            loss = -T.sum((y * T.log(T.clip(output, 1e-20, 1)) + (1 - y) * T.log(T.clip(1 - output, 1e-20, 1))), axis=1)

        if self.mode:
            loss = T.sum(loss)
        else:
            loss = T.mean(loss)
        return self.weight * loss


class MulticlassLogLoss(object):
    def __init__(self, weight=1., mode=0):
        self.weight = weight
        self.mode = mode
        if not (self.mode == 0 or self.mode == 1):
            raise ValueError("mode must be 0 or 1.")

    def get_output(self, y, output):
        # if categorical variables
        if y.ndim == 1:
            loss = -(T.log(1e-20 + output)[T.arange(y.shape[0]), y])
        # if one-hot
        elif y.ndim == 2:
            loss = -(T.sum(y * T.log(1e-20 + output), axis=1))
        # else
        else:
            raise Exception('Label Error:label must be scalar or vector. If not miss, you must rewrite model, objective etc.')

        if self.mode:
            loss = T.sum(loss)
        else:
            loss = T.mean(loss)
        return self.weight * loss


class SquaredError(object):
    def __init__(self, weight=1., mode=0):
        self.weight = weight
        self.mode = mode
        if not (self.mode == 0 or self.mode == 1):
            raise ValueError("mode must be 0 or 1.")

    def get_output(self, y, output):
        loss = (T.sum(T.square(y - output), axis=1))
        if self.mode:
            loss = T.sum(loss)
        else:
            loss = T.mean(loss)
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
