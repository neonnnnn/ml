from __future__ import absolute_import
import six
import sklearn


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


def shuffle(x, y, rng=1234):
    return sklearn.utils.shuffle(x, y, random_state=rng)
