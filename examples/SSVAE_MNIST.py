import numpy as np
import load_mnist
from ml.deeplearning.layers import Dense, Activation
from ml.deeplearning.normalizations import BatchNormalization
from ml.deeplearning.optimizers import Adam
from ml.deeplearning.models import Sequential
from ml.deeplearning.distributions import Gaussian, Bernoulli, Categorical
from ml.utils import BatchIterator
from SSVAE import SSVAE
from sklearn.metrics import accuracy_score
from ml.deeplearning import run_on_batch, variable
from ml.deeplearning.initializations import normal
import theano.tensor as T


def train_ssvae_mnist(X_l, y_l, X_u, X_valid, y_valid, dim,
                      z_dim, n_hidden, alpha, lr, activation='softplus',
                      epoch=3000, dis_decoder='bernoulli', sampler=None,
                      mode='analytical',
                      filename='m2_valid_acc.txt', decay=False):
    n_u = X_u.shape[0]
    n_l = X_l.shape[0]
    print(dim)
    rng = np.random.RandomState(1)
    encoder_y = Sequential(dim, rng)
    encoder_y.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    encoder_y.add(Activation(activation))

    if mode == 'mc':
        def annealing(temp, i):
            return T.switch(0.5 > T.exp(-3e-5 * i), 0.5, T.exp(-3e-5 * i))
    else:
        annealing = None

    encoder_y = Categorical(Dense(10, init=normal(0, 0.001)), encoder_y,
                            temp=1., annealing=annealing)
    encoder_z = Sequential(dim+10, rng)
    encoder_z.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    encoder_z.add(Activation(activation))
    encoder_z = Gaussian(mean_layer=Dense(z_dim, init=normal(0, 0.001)),
                         logvar_layer=Dense(z_dim, init=normal(0, 0.001)),
                         network=encoder_z)

    decoder = Sequential(z_dim+10, rng)
    decoder.add(BatchNormalization(Dense(n_hidden, init=normal(0, 0.001))))
    decoder.add(Activation(activation))
    if dis_decoder == 'bernoulli':
        decoder = Bernoulli(mean_layer=Dense(dim, init=normal(0, 0.001)),
                            network=decoder)
    elif dis_decoder == 'gaussian':
        decoder = Gaussian(mean_layer=Dense(dim, init=normal(0, 0.001)),
                           logvar_layer=Dense(dim, init=normal(0, 0.001)),
                           network=decoder)

    ssvae = SSVAE(rng, encoder_y=encoder_y, encoder_z=encoder_z,
                  decoder=decoder, mode=mode, alpha=alpha)
    opt = Adam(lr=lr)
    ssvae.compile(opt=opt, train_loss=None)

    pred_batches = BatchIterator([X_valid], 100)

    X_l_var = variable(X_l)
    X_u_var = variable(X_u)
    y_l_var = variable(y_l)
    function = ssvae.function(X_l_var, X_u_var, y_l_var, 'train', decay)
    test_function = ssvae.function(X_l_var, X_u_var, y_l_var, 'test')
    # training
    accs = []
    for i in xrange(epoch):
        loss = 0
        u_idxs = rng.permutation(n_u).reshape(n_u/100, 100)
        for u_idx in u_idxs:
            if sampler is not None:
                inputs = [sampler(X_l),
                          np.atleast_2d(y_l),
                          sampler(X_u[u_idx]),
                          ]
            else:
                inputs = [X_l,
                          np.atleast_2d(y_l),
                          X_u[u_idx],
                          ]
            loss += run_on_batch(inputs, function)[0]
        loss /= n_u/100.
        pred = np.argmax(encoder_y.predict(pred_batches), 1)
        accs += [accuracy_score(y_valid, pred)]
        print('epoch:{0} loss:{1} accuracy:{2}'.format(i+1, loss, accs[-1]))

    np.savetxt(filename, np.array(accs))
    return ssvae


def main():
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]

    n_l = 100
    X_l = np.zeros((n_l, 784), dtype=np.float32)
    y_l = np.zeros((n_l, 10), dtype=np.float32)
    idx = np.zeros(49900, dtype=np.int32)
    start = 0

    for i in range(10):
        idx_i = np.where(y_train == i)[0]
        X_l[i * n_l/10:(i + 1) * n_l/10] = X_train[idx_i[:n_l/10]]
        idx[start:start+len(idx_i[n_l/10:])] = idx_i[n_l/10:]
        y_l[i * n_l/10:(i + 1) * n_l/10, i] = 1.
        start += len(idx_i[n_l/10:])
    X_u = X_train[idx]
    epoch = 3000
    z_dim = 50
    n_hidden = 600
    filename = 'm2_valid_acc_mc.txt'
    rng = np.random.RandomState(1)

    def sampler(x):
        return rng.binomial(1, x).astype(np.float32)

    train_ssvae_mnist(X_l, y_l, X_u, X_valid, y_valid, 784,
                      z_dim, n_hidden, 1., 3e-4, 'softplus',
                      epoch, 'bernoulli', sampler, 'analytical',
                      filename)

if __name__ == '__main__':
    main()

