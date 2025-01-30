import tensorflow as tf
import keras
from keras import optimizers
from keras.models import Model
from keras.layers import (
    Dense,
)

from models.configurations import loss_functions
from models.configurations import encoders
from models.configurations import decoders


class VaeBase(Model):
    def __init__(self, config):
        super(VaeBase, self).__init__()
        self.config = config
        self.hyperparameters = config.hyperparameters

        self.encoder = getattr(encoders, config.network_components.encoder)(config)
        self.decoder = getattr(decoders, config.network_components.decoder)(config)

        self.loss_fn = getattr(loss_functions, config.network_components.loss_fnc)

        self.mu = Dense(units=config.hyperparameters.latent_dim, activation=None)
        self.log_var = Dense(units=config.hyperparameters.latent_dim, activation=None)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.total_val_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.recon_val_loss_tracker = keras.metrics.Mean(name="val_recon_loss")
        self.kl_val_loss_tracker = keras.metrics.Mean(name="val_kl_loss")

    def _reparameterize(self, mu, logvar, disable_noise):
        stddev = 1.0
        if disable_noise:
            stddev = 0.0

        eps = tf.random.normal(
            shape=mu.shape, mean=0.0, stddev=stddev, dtype=tf.float32
        )

        return eps * tf.exp(logvar * 0.5) + mu

    def encode(self, x):
        x_encoded = self.encoder(x)
        mu_encoded = self.mu(x_encoded)
        log_var_encoded = self.log_var(x_encoded)
        return mu_encoded, log_var_encoded

    def decode(self, z):
        return self.decoder(z)

    def call(self, x, disable_noise=False):
        mu_encoded, log_var_encoded = self.encode(x)
        z = self._reparameterize(mu_encoded, log_var_encoded, disable_noise)
        y_pred = self.decode(z)
        return y_pred, mu_encoded, log_var_encoded

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            total_loss, recon_loss, kl_loss = self.call_and_loss(x, y)
            self._apply_grads(total_loss, tape)
            return self._update_train_trackers(total_loss, recon_loss, kl_loss)

    def test_step(self, data):
        x, y = data
        total_loss, recon_loss, kl_loss = self.call_and_loss(x, y)
        return self._update_test_trackers(total_loss, recon_loss, kl_loss)

    def call_and_loss(self, x, y):
        y_pred, mu_encoded, log_var_encoded = self.call(x)
        total_loss, recon_loss, kl_loss = self.loss_fn(
            y,
            y_pred,
            mu_encoded,
            log_var_encoded,
            self.hyperparameters.alpha,
            self.hyperparameters.beta,
        )
        return total_loss, recon_loss, kl_loss

    def call_and_loss2(self, x, y):
        y_pred, mu_encoded, log_var_encoded = self.call(x)
        total_loss, recon_loss, kl_loss = self.loss_fn(
            y,
            y_pred,
            mu_encoded,
            log_var_encoded,
            self.hyperparameters.alpha,
            self.hyperparameters.beta,
        )
        return total_loss, recon_loss, kl_loss

    def _apply_grads(self, total_loss, tape):
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

    def _update_train_trackers(self, total_loss, recon_loss, kl_loss):
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def _update_test_trackers(self, total_loss, recon_loss, kl_loss):
        self.total_val_loss_tracker.update_state(total_loss)
        self.recon_val_loss_tracker.update_state(recon_loss)
        self.kl_val_loss_tracker.update_state(kl_loss)

        return {
            "val_total_loss": self.total_val_loss_tracker.result(),
            "val_recon_loss": self.recon_val_loss_tracker.result(),
            "val_kl_loss": self.kl_val_loss_tracker.result(),
        }

    def sample_z(self, mean=0.0, stddev=1.0):
        return tf.random.normal(
            shape=(self.hyperparameters.batch_size, self.hyperparameters.latent_dim),
            mean=mean,
            stddev=stddev,
        )

    def compile(self):
        super(VaeBase, self).compile(
            optimizer=optimizers.Adam(
                learning_rate=self.hyperparameters.learn_rate,
                clipnorm=self.hyperparameters.clipnorm,
            ),
            loss=self.loss_fn,
        )

    def synthesize_data(self, mean=0.0, stddev=1.0):
        z = self.sample_z(mean, stddev)
        return self.decode(z)
