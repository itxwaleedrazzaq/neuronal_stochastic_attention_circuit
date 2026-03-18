import tensorflow as tf
import math

class NSACLoss:
    def __init__(self, lambda_reg=0.5, epsilon=1e-3, name="NSAC-Loss"):
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon

    def __call__(self, y_true, y_pred):
        """
        y_pred = [id_samples, ood_samples]
        Each is a list of tuples (mu, std)
        """

        id_samples, ood_samples = y_pred

        # MC samples
        mu_id = tf.stack([p[0] for p in id_samples])
        std_id = tf.stack([p[1] for p in id_samples])
        mu_ood = tf.stack([p[0] for p in ood_samples])

        # NLL 
        var_id = tf.square(tf.exp(std_id)) + self.epsilon
        mu_mean = tf.reduce_mean(mu_id, axis=0)
        var_mean = tf.reduce_mean(var_id, axis=0)

        nll_loss = 0.5 * (
            tf.math.log(2.0 * math.pi) +
            tf.math.log(var_mean) +
            tf.square(y_true - mu_mean) / var_mean
        )
        nll_loss = tf.reduce_mean(nll_loss)

        # Epistemic regularizer
        epi_id = tf.reduce_mean(tf.math.reduce_variance(mu_id, axis=0))
        epi_ood = tf.reduce_mean(tf.math.reduce_variance(mu_ood, axis=0))
        reg_loss = tf.math.log(1.0 + (epi_id / (epi_ood + self.epsilon)))

        total_loss = nll_loss + self.lambda_reg * reg_loss

        return total_loss, nll_loss, reg_loss
    
class NLL:
    def __init__(self, epsilon=1e-2, name="gaussian_nll"):
        self.epsilon = epsilon
        
    def __call__(self, y_true, y_pred):
        mu, log_std = y_pred

        log_std = tf.clip_by_value(log_std, -10.0, 5.0)
        var = tf.exp(2.0 * log_std)

        nll = 0.5 * (
            tf.math.log(2.0 * math.pi) +
            tf.math.log(var) +
            tf.square(y_true - mu) / var
        )

        return tf.reduce_mean(nll)