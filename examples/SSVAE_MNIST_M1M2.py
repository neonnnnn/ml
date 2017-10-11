import numpy as np
import load_mnist
from ml.utils import BatchIterator
from ml.deeplearning import variable
from VAE_MNIST import train_vae_mnist
from SSVAE_MNIST import train_ssvae_mnist


def main():
    dataset = load_mnist.load_data()
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]

    print('training M1 model...')

    vae = train_vae_mnist(X_train, X_valid, z_dim=50, n_hidden=600, lr=3e-4,
                          activation='softplus', epoch=1000, batch_size=100)

    n_l = 100

    X_l = np.zeros((n_l, X_train.shape[1]), dtype=np.float32)
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
    epoch = 2000
    n_hidden = 500
    z_dim = 50
    dim = 50
    alpha = 1.
    mode = 'mc'
    filename = ('results/SSVAE/m1m2_valid_acc_' + str(mode) +'_alpha_'
                + str(alpha) +'_.txt')
    print('training M2 model...')

    encoder_function = vae.encoder.function(variable(X_l), mode='sampling')

    def sampler(x):
        return encoder_function(x)[0]

    X_valid = vae.encoder.predict(BatchIterator([X_valid], 100))
    train_ssvae_mnist(X_l, y_l, X_u, X_valid, y_valid, dim, z_dim, n_hidden,
                      alpha, 3e-4, 'softplus', epoch,
                      'gaussian', sampler, mode, filename)

if __name__ == '__main__':
    main()
