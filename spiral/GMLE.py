import tensorflow as tf
import numpy as np

class GMLE(tf.keras.Model):
    def __init__(self, model_fn, model_name='Gaussian-MLE'):
        super().__init__()
        self.model = model_fn
        self.model_name = model_name

        self.build((None, model_fn.input_shape[-1]))

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)
        return self.model(inputs, training=training)

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        """Standard Keras compile override."""
        if optimizer:
            self.optimizer = optimizer
        if loss:
            self.loss_fn = loss
        super().compile(metrics=metrics, **kwargs)

    def train_step(self, data):
        """Performs one training step with OOD-based aleatoric regularization."""
        x_id, y_id = data
        x_id = tf.cast(x_id, tf.float32)
        y_id = tf.cast(y_id, tf.float32)

        with tf.GradientTape() as tape:
            y_pred = self.model(x_id, training=True)
            total_loss = self.loss_fn(y_id, y_pred)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return {"loss": total_loss, "nll": total_loss}

    def test_step(self, data):
        x, y = data
        x_id = tf.cast(x, tf.float32)
        y_id = tf.cast(y, tf.float32)
        y_pred_id = self.model(x_id, training=False)
        total_loss = self.loss_fn(y_id, y_pred_id)

        return {"loss": total_loss, "nll": total_loss}

    
    def calculate_uncertainty(self, X):
        X = tf.cast(X, tf.float32)
        mu, log_std = self.model(X, training=False)
        aleatoric_unc = tf.exp(2.0 * log_std)  # variance = (exp(log_std))^2
        epistemic_unc = tf.zeros_like(aleatoric_unc)
        total_uncertainty = aleatoric_unc + epistemic_unc

        return mu, aleatoric_unc, epistemic_unc, total_uncertainty
    
    def predict(self, X, *kwargs):
        mean_pred, epistemic, aleatoric, total_uncertainty = self.calculate_uncertainty(X)
        return mean_pred.numpy(), total_uncertainty.numpy()

    def predict_with_uncertainty(self, X):
        mean_pred, epistemic, aleatoric, total_uncertainty = self.calculate_uncertainty(X)
        return mean_pred.numpy(), aleatoric.numpy(), epistemic.numpy()

    def comprehensive_evaluate(self, X_test, y_test):
        X_test = tf.cast(X_test, tf.float32)
        y_test = tf.cast(y_test, tf.float32)

        # ---- Per-sample prediction ----
        mu, log_std = self.model(X_test, training=False)
        std = tf.exp(log_std)

        # ---- Reduce to scalars ----
        mu_mean = tf.reduce_mean(mu).numpy()
        std_mean = tf.reduce_mean(std).numpy()
        mse_value = tf.reduce_mean(tf.square(y_test - mu)).numpy()

        # ---- Use built-in evaluation to get final NLL ----
        results = super().evaluate(
            X_test,
            y_test,
            return_dict=True,
            verbose=0
        )

        nll_value = float(results["nll"])

        return mu_mean, std_mean, nll_value, mse_value
