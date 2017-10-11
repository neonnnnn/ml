from GAN import GAN
import theano
import theano.tensor as T


class WGAN(GAN):
    def __init__(self, rng, generator, discriminator, clipping=(-0.01, 0.01)):
        self.clipping=clipping
        super(WGAN, self).__init__(rng, generator, discriminator)

    def function(self, x, mode='train'):
        z = T.matrix('z', dtype=theano.config.floatX)
        if mode == 'train':
            pred_real = self.discriminator(x, train=True)
            fake = self.generator(z, train=True)
            pred_fake = self.discriminator(fake, train=True)

            loss_gen = - T.sum(pred_fake) / pred_fake.shape[0]
            loss_dis = - (T.sum(pred_real) / pred_real.shape[0] - T.sum(pred_fake)/pred_fake.shape[0])
            updates_gen = self.opt_gen.get_updates(loss_gen, self.generator.params)
            updates_dis = self.opt_dis.get_updates(loss_dis, self.discriminator.params)
            updates_dis_clipped = []
            for update in updates_dis:
                updates_dis_clipped.append((update[0], T.clip(update[1], *self.clipping)))

            function_gen = theano.function(inputs=[z],
                                           outputs=[loss_gen],
                                           updates=updates_gen+self.updates)

            function_dis = theano.function(inputs=[x, z],
                                           outputs=[loss_dis],
                                           updates=updates_dis_clipped+self.get_updates())

            return function_gen, function_dis
        elif mode == 'pred':
            fake = self.generator(z, train=False)
            pred = self.discriminator(fake, train=False)
            function = theano.function(inputs=[z], outputs=[pred])

            return function