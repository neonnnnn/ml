import theano.tensor as T
import theano
from GAN import GAN
from ml.deeplearning.theanoutils import sharedasarray
import timeit
import numpy as np
from ml.utils import progbar


def L1(x, y, axis):
    return T.mean(T.sum(abs(x-y), axis=axis))


def L2(x, y, axis):
    return T.mean(T.sum((x-y)**2, axis=axis))


class BEGAN(GAN):
    def __init__(self, rng, generator, encoder, decoder, gamma=0.5, lam=0.001):
        self.function_gen = None
        self.function_dis = None
        self.opt_gen = None
        self.opt_dis = None
        self.gamma = gamma
        self.k = sharedasarray(0.)
        self.lam = lam
        super(GAN, self).__init__(rng, generator=generator, encoder=encoder, decoder=decoder)

    def function(self, x, z=None, mode='train'):
        if z is None:
            z = T.matrix('z', dtype=theano.config.floatX)
        if mode == 'train':
            pred_real = self.decoder(self.encoder(x, train=True), train=True)
            fake = self.generator(z, train=True)
            pred_fake = self.decoder(self.encoder(fake, train=True), train=True)
            axis = [i for i in range(1, fake.ndim)]
            L_real = L2(x, pred_real, axis)
            L_fake = L2(fake, pred_fake, axis)

            loss_gen = L_fake
            loss_dis = L_real - self.k * L_fake
            gen_params = self.generator.params
            dis_params = self.encoder.params+self.decoder.params
            updates_gen = self.opt_gen.get_updates(loss_gen, gen_params)
            updates_dis = self.opt_dis.get_updates(loss_dis, dis_params)

            kt = self.k + self.lam*(self.gamma * L_real - L_fake)
            updates_gen.append((self.k, T.clip(kt, 0., 1.)))
            function_dis = theano.function(inputs=[x, z],
                                           outputs=[loss_dis],
                                           updates=updates_dis + self.get_updates())
            function_gen = theano.function(inputs=[x, z],
                                           outputs=[loss_gen],
                                           updates=updates_gen+self.updates)

            return function_gen, function_dis
        elif mode == 'pred':
            fake = self.generator(z, train=False)
            pred = self.discriminator(fake, train=False)
            function = theano.function(inputs=[z], outputs=[pred])

            return function
        elif mode == 'test':
            pred_real = self.decoder(self.encoder(x, train=False), train=False)
            fake = self.generator(z, train=False)
            pred_fake = self.decoder(self.encoder(fake, train=False), train=False)
            axis = [i for i in range(1, fake.ndim)]
            L_real = L2(x, pred_real, axis)
            L_fake = L2(fake, pred_fake, axis)

            M_global = L_real + abs(self.gamma * L_real - L_fake)
            function = theano.function(inputs=[x, z], outputs=[M_global])

            return function

    def _fit(self, train_batches, epoch, iprint=True, k=1, valid_batches=None):
        if iprint:
            start_time = timeit.default_timer()
        # training while i < epoch
        loss_gen_all = []
        loss_dis_all = []
        n_batches = train_batches.n_batches
        loss_valid = None
        for i in xrange(epoch):
            start_epoch = timeit.default_timer()
            loss_gen = []
            loss_dis = []

            for j, batch in enumerate(train_batches):
                z = np.random.standard_normal((batch[0].shape[0], self.generator.n_in))
                loss_dis += self.function_dis(batch[0], z.astype(np.float32))
                if (j+1) % k == 0:
                    z = np.random.standard_normal((batch[0].shape[0], self.generator.n_in))
                    loss_gen += self.function_gen(batch[0], z.astype(np.float32))
                if iprint:
                    e = timeit.default_timer()
                    progbar(j + 1, n_batches, e - start_epoch)
            loss_gen_all += [np.mean(loss_gen)]
            loss_dis_all += [np.mean(loss_dis)]

            if iprint:
                if valid_batches is not None:
                    loss_valid = self.test(valid_batches)
                time = timeit.default_timer() - start_epoch
                self._print(i, loss_dis_all[-1], loss_gen_all[-1], time, loss_valid)

        # training end
        end_time = timeit.default_timer()
        if iprint:
            print(' ran for {0:.2f}m'.format((end_time - start_time)/60.))

        return loss_dis_all, loss_gen_all
