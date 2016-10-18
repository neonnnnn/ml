from __future__ import absolute_import
import six
import numpy as np
import sklearn
import sys
import matplotlib.pyplot as plt
from scipy.misc import imsave


class BatchIterator(object):
    def __init__(self, data, batch_size, shuffle=True, seed=1234):
        self._data = data
        self._current = 0
        self.batch_size = batch_size

        if isinstance(data, (tuple, list)):
            self.n_samples = len(data[0])
            self._data_cate = len(data)
            if self._data_cate == 1:
                self._data = np.array(self._data[0])
        else:
            self.n_samples = len(data)
            self._data_cate = 1

        if batch_size > self.n_samples:
            raise ValueError("Invalid Batch size. Batch size must be <= n_samples")

        self.n_batches = self.n_samples / self.batch_size
        if self.n_samples % self.batch_size != 0:
            self.n_batches += 1

        if shuffle:
            self._rng = np.random.RandomState(seed)
            self._idx = self._rng.permutation(self.n_samples)
        else:
            self.rng = None
            self._idx = None

    def __iter__(self):
        return self

    def next(self):
        if self._current > (self.n_batches - 1):
            self._current = 0
            raise StopIteration

        if self._data_cate == 1:
            batch = [self._make_batch(self._data)]
        else:
            batch = map(lambda x: self._make_batch(x), self._data)

        self._current += 1

        return batch

    def _make_batch(self, data):
        if self._current == self.n_batches - 1:
            if self._idx is None:
                batch = data[-self.batch_size:]
            else:
                batch = data[self._idx[-self.batch_size:]]
        else:
            start = self._current * self.batch_size
            end = start + self.batch_size
            if self._idx is None:
                batch = data[start:end]
            else:
                batch = data[self._idx[start:end]]
        return batch


def num_of_error(y, p_y_given_x):
    if p_y_given_x.ndim != 1:
        y_pred = np.argmax(p_y_given_x, axis=1)
    else:
        y_pred = (np.sign(p_y_given_x - 0.5) + 1) // 2
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


def onehot(y):
    max_idx = np.max(y)
    onehot_y = np.zeros((len(y), max_idx), dtype=np.int32)
    onehot_y[np.arange(len(y)), y] = 1
    return onehot_y


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


def progbar(now, max_value, time=None):
    width = int(30 * now/max_value)
    prog = "[%s]" % ("=" * width + ">" + " " * (30 - width))
    if now != max_value and time is not None:
        eta = time * (max_value - now) / now
        sys.stdout.write("\r{0}{1}/{2}, eta:{3:.2f}s".format(prog, now, max_value, eta))
        sys.stdout.flush()
    else:
        sys.stdout.write("\r{0}{1}/{2}, {3:.2f}s".format(prog, now, max_value, time))
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
    return plt


def saveimg(data, figshape, filename):
    h, w = data[0].shape[:2]
    img = np.zeros((h*figshape[0], w*figshape[1]))
    for n, x in enumerate(data):
        j = n / figshape[0]
        i = n % figshape[1]
        img[j*h:j*h+h, i*w:i*w+w] = x
    if filename is not None:
        imsave(filename, img)
    return img


def color_saveimg(data, (nh, nw), save_path=None):
    c, h, w = data[0].shape[:]
    img = np.zeros((h*nh, w*nw, c))
    for n, x in enumerate(data):
        j = n / nw
        i = n % nw
        img[j*h:j*h+h, i*w:i*w+w, :] = x.transpose(1, 2, 0)
    if save_path is not None:
        imsave(save_path, img)
    return img
