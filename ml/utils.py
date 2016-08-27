from __future__ import absolute_import
import six
import numpy as np
import sklearn
import sys
import matplotlib.pyplot as plt


def num_of_error(y, p_y_given_x):
    y_pred = np.argmax(p_y_given_x, axis=1)
    # check if y has same dimension of y_pred
    if y.ndim != y_pred.ndim:
        raise TypeError(
            'y should have the same shape as self.y_pred',
            ('y', y.type, 'y_pred', y_pred.type)
        )
    a = y - y_pred
    return len(np.where(a != 0)[0])


def get_from_module(identifier, module_params, module_name, instantiate=False, kwargs=None):
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            raise Exception('Invalid ' + str(module_name) + ': ' + str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res

    return identifier


def reshape_img(data, imshape):
    if imshape[1] * imshape[2] != (data.shape[1] / imshape[0]):
        raise Exception('height or width or channel is wrong. (data.shape[1]/channel) must be equal to (height * weight).')
    else:
        return np.reshape(data, (data.shape[0], imshape[0], imshape[1], imshape[2]))


def shuffle(x, y):
    return sklearn.utils.shuffle(x, y, random_state=1234)


def make_validation(x, y, validation_rate):
    x, y = shuffle(x, y)
    size = x.shape[0]
    valid_x = x[0:int(size * validation_rate)]
    valid_y = y[0:int(size * validation_rate)]
    x = x[int(size * validation_rate):]
    y = y[int(size * validation_rate):]

    return x, y, valid_x, valid_y


def progbar(now, max_value):
    width = int(30 * now/max_value)
    prog = "[%s]" % ("=" * width + ">" + " " * (30 - width))
    if now == max:
        sys.stdout.write(prog + str(now) + "/" + str(max_value))
        sys.stdout.flush()
        sys.std.write("\n")
    else:
        sys.stdout.write("\r" + prog + str(now) + "/" + str(max_value))
        sys.stdout.flush()


def visualize(data, figshape, filename, nomarlization_flag=True):

    if nomarlization_flag:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data *= 255.0
    data = data.astype(np.int)
    pos = 1
    for i in xrange(figshape[0]*figshape[1]):
        plt.subplot(figshape[0], figshape[0], pos)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(data[i])
        plt.gray()
        plt.axis('off')
        pos += 1
    plt.savefig(filename)



