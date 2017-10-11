import theano.tensor as T
import theano
from ml.deeplearning.objectives import CrossEntropy
from ml.deeplearning.models import Model
from ml.deeplearning.theanoutils import variable, run, run_on_batch
import timeit
import sys
import numpy as np
from ml.utils import progbar


class GAN(Model):
    def __init__(self, rng, generator, discriminator):
        self.function_gen = None
        self.function_dis = None
        self.opt_gen = None
        self.opt_dis = None
        super(GAN, self).__init__(rng, generator=generator, discriminator=discriminator)

    def forward(self, x, train):
        pass

    def function(self, x, mode='train'):
        z = T.matrix('z', dtype=theano.config.floatX)
        if mode == 'train':
            pred_real = self.discriminator(x, train=True)
            fake = self.generator(z, train=True)
            pred_fake = self.discriminator(fake, train=True)
            ce = CrossEntropy(mode='mean')

            y_one = T.ones(shape=pred_real.shape, dtype=theano.config.floatX)
            y_zero = T.zeros(shape=pred_real.shape, dtype=theano.config.floatX)
            y_one_fake = T.ones(shape=pred_fake.shape, dtype=theano.config.floatX)

            loss_gen = ce.calc(y_one_fake, pred_fake)
            loss_dis = ce.calc(y_one, pred_real) + ce.calc(y_zero, pred_fake)
            updates_gen = self.opt_gen.get_updates(loss_gen, self.generator.params)
            updates_dis = self.opt_dis.get_updates(loss_dis, self.discriminator.params)
            function_dis = theano.function(inputs=[x, z],
                                           outputs=[loss_dis],
                                           updates=updates_dis+self.get_updates())
            function_gen = theano.function(inputs=[z],
                                           outputs=[loss_gen],
                                           updates=updates_gen+self.updates)

            return function_gen, function_dis
        elif mode == 'pred':
            fake = self.generator(z, train=False)
            pred = self.discriminator(fake, train=False)
            function = theano.function(inputs=[z], outputs=[pred])

            return function

    def compile(self, opt_gen, opt_dis):
        self.opt_gen = opt_gen
        self.opt_dis = opt_dis
        self.train_function = None
        self.test_function = None

    def fit(self, train_batches, epoch=100,  valid_batches=None, iprint=True, k=1):
        variables = [variable(datum) for datum in train_batches.data]
        if self.function_gen is None or self.function_dis is None:
            self.function_gen, self.function_dis = self.function(*variables, mode='train')

        if iprint:
            header = 'epoch' + ' '*5 + 'D Loss' + ' '*7 + 'G Loss'
            if valid_batches is not None:
                header += ' '*7 + 'valid_loss'
                header += ' '*3 + 'time'
            else:
                header += ' '*7 + 'time'

            print(header)

        return self._fit(train_batches, epoch, iprint, k, valid_batches)

    def _print(self, i, loss_dis, loss_gen, time, loss_valid=None):
        n_space = 10 - len(str(i + 1))
        sys.stdout.write('{0}{1}'.format(i + 1, ' ' * n_space))
        n_space = 13 - len('{0:.5f}'.format(loss_dis))
        sys.stdout.write('{0:.5f}{1}'.format(loss_dis, ' ' * n_space))
        n_space = 13 - len('{0:.5f}'.format(loss_gen))
        sys.stdout.write('{0:.5f}{1}'.format(loss_gen, ' ' * n_space))
        if loss_valid is not None:
            n_space = 13 - len('{0:.5f}'.format(loss_valid))
            sys.stdout.write('{0:.5f}{1}'.format(loss_valid, ' ' * n_space))
        sys.stdout.write('{0:.2f}s\n'.format(time))

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
                    loss_gen += self.function_gen(z.astype(np.float32))
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
