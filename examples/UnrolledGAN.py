import theano.tensor as T
import theano
from GAN import GAN
import timeit
import numpy as np
from ml.utils import progbar
import sys


class UnrolledGAN(GAN):
    def __init__(self, rng, generator, discriminator, K=10):
        self.function_gen = None
        self.function_dis = None
        self.opt_gen = None
        self.opt_dis = None
        self.K = K
        super(UnrolledGAN, self).__init__(rng, generator=generator, discriminator=discriminator)

    def _fit(self, train_batches, epoch, iprint=True, k=1):
        if iprint:
            start_time = timeit.default_timer()
        # training while i < epoch
        loss_gen_all = []
        loss_dis_all = []
        n_batches = train_batches.n_batches
        for i in xrange(epoch):
            start_epoch = timeit.default_timer()
            loss_gen = []
            loss_dis = []

            for j, batch in enumerate(train_batches):
                params = []
                # unrolling
                for _ in xrange(self.K):
                    z = np.random.standard_normal((batch[0].shape[0], self.generator.n_in))
                    loss_dis += self.function_dis(batch[0], z.astype(np.float32))
                    if not len(params):
                        for param in self.discriminator.params:
                            params.append([p.get_value() for p in param])
                z = np.random.standard_normal((batch[0].shape[0], self.generator.n_in))
                loss_gen += self.function_gen(z.astype(np.float32))
                if iprint:
                    e = timeit.default_timer()
                    progbar(j + 1, n_batches, e - start_epoch)

                # setting params for discriminator
                for param, stored in self.discriminator.params, params:
                    for p, s in param, stored:
                        p.set_value(s)

            loss_gen_all += [np.mean(loss_gen)]
            loss_dis_all += [np.mean(loss_dis)]

            if iprint:
                self._print(i, loss_dis_all[-1], loss_gen_all[-1])

            # if there are valid data, calc valid_error
            if iprint:
                end_epoch = timeit.default_timer()
                sys.stdout.write('{0:.2f}s\n'.format(end_epoch - start_epoch))

        # training end
        end_time = timeit.default_timer()
        if iprint:
            print(' ran for {0:.2f}m'.format((end_time - start_time) / 60.))

        return loss_dis_all, loss_gen_all