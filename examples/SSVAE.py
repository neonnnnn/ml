import theano.tensor as T
import theano
from ml.deeplearning.objectives import MulticlassLogLoss
from ml.deeplearning.models import Model
from VAE import KLD
from ml.deeplearning.distributions import Distribution
from ml.deeplearning.theanoutils import sharedasarray


class SSVAE(Model):
    def __init__(self, rng, encoder_y, encoder_z, decoder, mode='analytical', alpha=1., Nmc=1.):
        self.mode = mode
        self.alpha = alpha
        self.Nmc = Nmc
        if not isinstance(encoder_y, Distribution):
            raise TypeError('Encoder must be Distribution.')
        if not isinstance(decoder, Distribution):
            raise TypeError('Decoder must be Distribution.')
        if not isinstance(encoder_z, Distribution):
            raise TypeError('Encoder must be Distribution.')
        super(SSVAE, self).__init__(rng, encoder_y=encoder_y,
                                    encoder_z=encoder_z, decoder=decoder)

    def forward(self, X_l, y, X_u, train):
        if train:
            # about labelled data
            Xy = T.concatenate([X_l, y], axis=1)
            z_mean, z_var, z = self.encoder_z(Xy, sampling=True, train=train)
            zy = T.concatenate([z, y], axis=1)
            decoder_l = self.decoder(zy, sampling=False, train=train)
            pred_y = self.encoder_y(X_l, train=train)

            if isinstance(decoder_l, (tuple, list)):
                log_likelihood_l = self.decoder.log_likelihood(X_l, *decoder_l)
            else:
                log_likelihood_l = self.decoder.log_likelihood(X_l, decoder_l)

            # about unlabelled data
            if self.mode == 'analytical':
                y_mean_u = self.encoder_y(X_u, train=train)
                y_u = T.tile(T.eye(self.encoder_y.n_out), (X_u.shape[0], 1))
                X_u = T.repeat(X_u, self.encoder_y.n_out, axis=0)
                Xy_u = T.concatenate([X_u, y_u], axis=1)
            else:
                # gumbel-softmax sampling
                y_mean_u, y_u = self.encoder_y(X_u, sampling=True, train=train)
                Xy_u = T.concatenate([X_u, y_u], axis=1)

            z_mean_u, z_var_u, z_u = self.encoder_z(Xy_u, True, train=train)
            zy_u = T.concatenate([z_u, y_u], axis=1)
            decoder_u = self.decoder(zy_u, sampling=False, train=train)
            if isinstance(decoder_u, (tuple, list)):
                log_likelihood_u = self.decoder.log_likelihood(X_u, *decoder_u)
            else:
                log_likelihood_u = self.decoder.log_likelihood(X_u, decoder_u)

            return (z_mean, z_var, pred_y, log_likelihood_l, decoder_l,
                    z_mean_u, z_var_u, y_mean_u, log_likelihood_u)

    def function(self, X_l, y, X_u, mode='train', decay=False):
        n_l = X_l.shape[0]
        n_u = X_u.shape[0]
        (z_mean, z_var, pred_y, log_likelihood_l, decoder_l,
         z_mean_u, z_var_u, y_mean_u, log_likelihood_u) \
            = self.forward(X_l, y, X_u, train=True)

        kld_z = KLD().calc(z_mean, z_var)
        pred_loss = MulticlassLogLoss().calc(y, pred_y)
        entropy = T.sum(y_mean_u * T.log(y_mean_u+1e-10), axis=1)
        kld_z_u = KLD().calc(z_mean_u, z_var_u)

        if self.mode == 'analytical':
            kld_z_u *= T.flatten(y_mean_u)
            log_likelihood_u *= T.flatten(y_mean_u)

        loss = (T.sum(kld_z - log_likelihood_l)/n_l
                + T.sum(kld_z_u - log_likelihood_u)/n_u + T.mean(entropy))

        if isinstance(self.alpha, (int, float)):
            self.alpha = sharedasarray(self.alpha)
            loss += self.alpha*pred_loss
            loss /= (1+self.alpha)

        if mode == 'train':
            updates = self.opt.get_updates(T.mean(loss), self.params)
            updates += self.get_updates()

            if decay and self.alpha is not False:
                updates.append((self.alpha, 0.99999 * self.alpha))

            return theano.function(inputs=[X_l, y, X_u], outputs=[loss],
                                   updates=updates)
        else:
            return theano.function(inputs=[X_l, y, X_u],
                                   outputs=[decoder_l[0], decoder_l[1],
                                            T.sum(- log_likelihood_l)/n_l,
                                            pred_loss,
                                            T.sum(-log_likelihood_u)/n_u])
