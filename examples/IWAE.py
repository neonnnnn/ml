import theano.tensor as T
import theano
from VAE import VAE
from ml.deeplearning.theanoutils import log_mean_exp, log_sum_exp


class IWAE(VAE):
    def __init__(self, rng, encoder, decoder, k=5):
        self.k = k
        super(IWAE, self).__init__(rng, encoder=encoder, decoder=decoder)

    def forward(self, x, train):
        if train:
            rep_x = T.repeat(x, self.k, axis=0)
            q_mean, q_logvar, z = self.encoder.forward(rep_x, sampling=True, train=train)
            log_likelihood_q = self.encoder.log_likelihood_with_forward(z, rep_x, train=train)
            log_likelihood_p = self.decoder.log_likelihood_with_forward(rep_x, z)
            log_likelihood_p += self.encoder.log_likelihood(z, T.zeros(z.shape), T.zeros(z.shape))
            return log_likelihood_q, log_likelihood_p
        else:
            q_mean, q_logvar = self.encoder.forward(x, train=train)
            p = self.decoder.forward(q_mean, train=train)
            return q_mean, q_logvar, p

    def function(self, x, mode='train'):
        N = x.shape[0]
        if mode == 'train':

            log_likelihood_q, log_likelihood_p = self.forward(x, True)
            log_w = (log_likelihood_p - log_likelihood_q).reshape((N, self.k))
            # computing loss straightforward (Eq 8)
            loss = -T.mean(log_mean_exp(log_w, axis=1))

            """
            # computing loss for gradient(Eq 13)
            # this is slow
            w = T.exp(log_w - T.max(log_w, axis=1, keepdims=True))
            w_tilde = w / T.sum(w, axis=1, keepdims=True)
            dummy = T.matrix(dtype=theano.config.floatX)
            loss = -theano.clone(output=T.sum(dummy * log_w) / N, replace={dummy: w_tilde})
            """

            updates = self.opt.get_updates(loss, self.params)
            updates += self.get_updates()
            return theano.function(inputs=[x], outputs=[loss], updates=updates)
        else:
            return super(IWAE, self).function(x, mode)



