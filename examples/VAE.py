import theano.tensor as T
import theano
from ml.deeplearning.objectives import CrossEntropy, Regularization
from ml.deeplearning.models import Model
from ml.deeplearning.distributions import Distribution


class KLD(Regularization):
    def __init__(self):
        super(KLD, self).__init__(weight=1.0)

    def calc(self, mean, logvar, var=None):
        axis = tuple(range(mean.ndim))[1:]
        output = - 0.5 * T.sum(1 + logvar - T.exp(logvar) - mean**2, axis=axis)

        return output


class VAE(Model):
    def __init__(self, rng, encoder, decoder):
        if not isinstance(encoder, Distribution):
            raise TypeError('Encoder must be Distribution.')
        if not isinstance(decoder, Distribution):
            raise TypeError('Decoder must be Distribution.')
        super(VAE, self).__init__(rng, encoder=encoder, decoder=decoder)

    def forward(self, x, train):
        if train:
            q_mean, q_logvar, z = self.encoder.forward(x, sampling=True, train=train)
            decoder = self.decoder.forward(z, sampling=False, train=train)
            if isinstance(decoder, (tuple, list)):
                log_likelihood = self.decoder.log_likelihood(x, *decoder)
            else:
                log_likelihood = self.decoder.log_likelihood(x, decoder)
            return q_mean, q_logvar, log_likelihood
        else:
            q_mean, q_logvar = self.encoder.forward(x, train=train)
            p = self.decoder.forward(q_mean, train=train)
            return q_mean, q_logvar, p

    def function(self, x, mode='train'):
        if mode == 'train':
            q_mean, q_logvar, log_likelihood = self.forward(x, train=True)
            kld = KLD().calc(q_mean, q_logvar)
            loss = -log_likelihood + kld
            updates = self.opt.get_updates(T.mean(loss), self.params)
            updates += self.get_updates()
            return theano.function(inputs=[x], outputs=[loss], updates=updates)
        elif mode == 'pred':
            q_mean, q_logvar, p = self.forward(x, train=False)
            if isinstance(p, (tuple, list)):
                function = theano.function(inputs=[x], outputs=list(p))
            else:
                function = theano.function(inputs=[x], outputs=[p])

            return function
        elif mode == 'test':
            q_mean, q_logvar, p = self.forward(x, train=False)
            log_likelihood = self.decoder.log_likelihood(x, *list(p))
            kld = KLD().calc(q_mean, q_logvar)
            loss = T.mean(-log_likelihood + kld)
            function = theano.function(inputs=[x], outputs=[loss])

        return function
