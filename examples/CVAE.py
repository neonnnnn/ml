import theano.tensor as T
import theano
from VAE import VAE, KLD


class CVAE(VAE):
    def forward(self, x, w, train):
        if train:
            q_mean, q_var, z = self.encoder.forward(x, w, sampling=True, train=train)
            decoded = self.decoder.forward(z, w, sampling=False, train=train)

            # e.g., distribution of p is Gaussian
            if isinstance(decoded, (tuple, list)):
                log_likelihood = self.decoder.log_likelihood(x, *decoded)
            # bernoulli
            else:
                log_likelihood = self.decoder.log_likelihood(x, decoded)
            return q_mean, q_var, log_likelihood
        else:
            q_mean, q_var = self.encoder.forward(x, w, train=train)
            p = self.decoder.forward(q_mean, train=train)
            return q_mean, q_var, p

    def function(self, x, w, mode='train'):
        if mode == 'train':
            q_mean, q_var, log_likelihood = self.forward(x, w, train=True)
            kld = KLD().calc(q_mean, q_var)
            loss = -log_likelihood + kld
            updates = self.opt.get_updates(T.mean(loss), self.params)
            updates += self.get_updates()
            return theano.function(inputs=[x, w], outputs=[loss], updates=updates)
        elif mode == 'pred':
            q_mean, q_var, p = self.forward(x, train=False)
            if isinstance(p, (tuple, list)):
                function = theano.function(inputs=[x, w], outputs=list(p))
            else:
                function = theano.function(inputs=[x, w], outputs=[p])

            return function
        elif mode == 'test':
            q_mean, q_var, p = self.forward(x, w, train=False)
            log_likelihood = self.decoder.log_likelihood(x, *list(p))
            kld = KLD().calc(q_mean, q_var)
            loss = T.mean(-log_likelihood + kld)
            function = theano.function(inputs=[x, w], outputs=[loss])

        return function
