import h5py
import json
from theano.compile.sharedvalue import SharedVariable
import numpy as np
from models import Sequential, Model
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
    f.attrs['config'] = json.dumps(model.get_config()).encode('utf8')
    if isinstance(model, Model):
        rng_group = f.create_group('rng')
        state = model.rng.get_state()
        rng_group.attrs['pos'] = state[2]
        rng_group.attrs['has_gauss'] = state[3]
        rng_group.attrs['cached_gaussian'] = state[4]
        _create_group(rng_group, 'keys', state[1])

    if hasattr(model, 'get_layers_with_names_configs'):
        layers, names, config = model.get_layers_with_names_configs()
        layers_group = f.create_group('Layers')
        layers_group.attrs['config'] = json.dumps(config)
        for layer, name in zip(layers, names):
            if hasattr(layer, 'params'):
                g = layers_group.create_group(name)
                _save(layer, g)

    for k, v in model.__dict__.items():
        if isinstance(v, SharedVariable):
            val = v.get_value()
            _create_group(f, k, val)


def load(f):
    if isinstance(f, str):
        f = h5py.File(f, 'r')

    model = _load(f)

    f.close()

    return model


def _load(f):
    attrs = f.attrs
    config = json.loads(attrs['config'], object_pairs_hook=OrderedDict)
    if 'class' in config.key():
        model_class = config.pop('class')
    for key in f.keys():
        if key=='rng':
            rng = np.random.RandomState()
            state = ['MIT19937',
                     f['rng']['keys'].value,
                     f['rng']['pos'],
                     f['rng']['has_gaussian'],
                     f['rng']['pos']]
            rng.set_state(tuple(state))
            config.update({'rng': rng})
        if key == 'Layers':
            # OrderedDict
            layers = _load_layers(f['Layers'])
        # parameters
        else:
            config.update({key: f[key].value})

    if model_class == 'Sequential':
        model = globals()[model_class](config)
        for layer in layers.items():
            model.add(layer)
    else:
        config.update(layers)
        model = globals()[model_class](config)

    return model


def _load_layers(f):
    layers = OrderedDict()
    configs = json.loads(f.attrs['config'], object_pairs_hook=OrderedDict)
    for key, value in configs:
        if key in f.keys():
            layer = _load(f[key])
        else:
            layer_class = value.pop('class')
            layer = globals()[layer_class](value)

        layers.update({key:layer})
    return layers