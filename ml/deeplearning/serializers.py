from __future__ import absolute_import
import h5py
import json
from theano.compile.sharedvalue import SharedVariable
import numpy as np
from .models import Sequential, Model
from .distributions import *
from .normalizations import *
from .layers import *
from collections import OrderedDict


def _create_group(f, name, val):
    dset = f.create_dataset(name, val.shape, val.dtype)
    if not val.shape:
        dset[()] = val
    else:
        dset[:] = val


def save(model, f):
    if isinstance(f, str):
        f = h5py.File(f, 'w')

    _save(model, f)

    f.flush()
    f.close()


def _save(model, f):
    f.attrs['name'] = model.__class__.__name__
    f.attrs['config'] = json.dumps(model.get_config(), default=lambda x: None).encode('utf-8')
    if isinstance(model, Model):
        rng_group = f.create_group('rng')
        state = model.rng.get_state()
        rng_group.attrs['pos'] = state[2]
        rng_group.attrs['has_gauss'] = state[3]
        rng_group.attrs['cached_gaussian'] = state[4]
        _create_group(rng_group, 'keys', state[1])

    if hasattr(model, 'get_layers_with_names_configs'):
        layers, names, config = model.get_layers_with_names_configs()
        if layers is not None:
            layers_group = f.create_group('Layers')
            layers_group.attrs['config'] = json.dumps(config, default=lambda x: None).encode('utf-8')
            for layer, name in zip(layers, names):
                if hasattr(layer, 'params'):
                    g = layers_group.create_group(name)
                    _save(layer, g)

    for k, v in model.__dict__.items():
        if isinstance(v, SharedVariable):
            val = v.get_value()
            _create_group(f, k, val)


def load(f, layers_dic=globals()):
    if isinstance(f, str):
        f = h5py.File(f, 'r')
    model = _load(f, layers_dic)

    f.close()

    return model


def _load(f, layers_dic):
    attrs = f.attrs
    config = json.loads(attrs['config'].decode('utf-8'), object_pairs_hook=OrderedDict)
    if 'class' in config.keys():
        model_class = config.pop('class')
    layers = None
    for key in f.keys():
        if key == 'rng':
            rng = np.random.RandomState()
            state = ['MT19937',
                     f['rng']['keys'].value,
                     f['rng'].attrs['pos'],
                     f['rng'].attrs['has_gauss'],
                     f['rng'].attrs['cached_gaussian']]
            rng.set_state(tuple(state))
            config.update({'rng': rng})
        elif key == 'Layers':
            # OrderedDict
            layers = _load_layers(f['Layers'], layers_dic)
        # parameters (Dataset)
        else:
            config.update({key: f[key].value})

    if model_class == 'Sequential':
        model = layers_dic[model_class](**config)
        for layer in layers.values():
            model.add(layer)
    else:
        if layers is not None:
            config.update(layers)
        model = layers_dic[model_class](**config)
    return model


def _load_layers(f, layers_dic):
    layers = OrderedDict()
    configs = json.loads(f.attrs['config'].decode('utf-8'), object_pairs_hook=OrderedDict)
    for key, value in configs.items():
        # if layer has params:
        if key in f.keys():
            layer = _load(f[key], layers_dic)
        else:
            layer_class = value.pop('class')
            layer = layers_dic[layer_class](**value)

        layers.update({key: layer})
    return layers