import theano.tensor as T
import theano
from GAN import GAN


def L1(x, y, axis):
    return T.sum(abs(x-y), axis=axis)


def L2(x, y, axis):
    return T.sum((x-y)**2, axis=axis)


class EBGAN(GAN):
    def __init__(self, rng, generator, encoder, decoder, m=8., use_PT=False):
        self.function_gen = None
        self.function_dis = None
        self.opt_gen = None
        self.opt_dis = None
        self.m = m
        self.use_PT = use_PT
        super(GAN, self).__init__(rng, generator=generator, encoder=encoder, decoder=decoder)

    def function(self, x, mode='train'):
        z = T.matrix('z', dtype=theano.config.floatX)
        if mode == 'train':
            reconst_real = self.decoder(self.encoder(x, train=True), train=True)
            fake = self.generator(z, train=True)
            enc_fake = self.encoder(fake, train=True)
            reconst_fake = self.decoder(enc_fake, train=True)

            axis = [i for i in range(1, fake.ndim)]
            loss_gen = T.mean(L2(fake, reconst_fake, axis))
            if self.use_PT:
                N = enc_fake.shape[0]
                PT = (T.dot(enc_fake, enc_fake.T) ** 2
                      / T.dot(T.sum(enc_fake ** 2, axis=1, keepdims=True),
                              T.sum(enc_fake ** 2, axis=1, keepdims=True).T)
                      )

                PT = (T.sum(PT) - T.nlinalg.trace(PT)) / (N * (N-1))
                loss_gen += PT
            loss_dis = T.mean(L2(x, reconst_real, axis)
                              + T.maximum(0, self.m - L2(fake, reconst_fake, axis)))
            gen_params = self.generator.params
            dis_params = self.encoder.params+self.decoder.params
            updates_gen = self.opt_gen.get_updates(loss_gen, gen_params)
            updates_dis = self.opt_dis.get_updates(loss_dis, dis_params)
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