import tensorflow as tf

class Evidential(tf.keras.Model):
    def __init__(self, model_fn, lambda_reg, model_name='Evidential'):
        super().__init__()
        self.model = model_fn
        self.model_name = model_name
        self.lambda_reg = lambda_reg

        self.build((None, model_fn.input_shape[-1]))

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)
        return self.model(inputs, training=training)

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def predict(self, X):
        X = tf.cast(X, tf.float32)
        output = self.model(X, training=False)
        mu, v, alpha, beta = tf.split(output, 4, axis=-1)

        eps = 1e-8
        var = beta / (alpha - 1 + eps) + beta / (v * (alpha - 1 + eps))
        std = tf.sqrt(var + eps)

        return tf.concat([mu, std], axis=-1).numpy()
    
    def calculate_uncertainty(self,X):
        X = tf.cast(X, tf.float32)
        output = self.model(X, training=False)
        mu, v, alpha, beta = tf.split(output, 4, axis=-1)

        eps = 1e-8
        aleatoric = beta / (alpha - 1 + eps)
        epistemic = beta / (v * (alpha - 1 + eps))
        
        return mu, aleatoric, epistemic
    
        

    def predict_with_uncertainty(self, X):
        X = tf.cast(X, tf.float32)
        output = self.model(X, training=False)
        mu, v, alpha, beta = tf.split(output, 4, axis=-1)

        eps = 1e-8
        aleatoric = beta / (alpha - 1 + eps)
        epistemic = beta / (v * (alpha - 1 + eps))

        return mu.numpy(), aleatoric.numpy(), epistemic.numpy()
    
    def comprehensive_evaluate(self, X_test, y_test):

        X_test = tf.cast(X_test, tf.float32)
        y_test = tf.cast(y_test, tf.float32)

        # --- Uncertainty Decomposition ---
        mean_pred, aleatoric, epistemic = self.calculate_uncertainty(X_test)

        total_variance = aleatoric + epistemic
        mean_pred = tf.reduce_mean(mean_pred)
        total_variance = tf.reduce_mean(total_variance)

        predictive_std = tf.sqrt(total_variance) + 1e-8

        # ---- Use built-in evaluation to get final NLL ----
        results = super().evaluate(
            X_test,
            y_test,
            return_dict=True,
            verbose=0
        )

        nll_value = float(results["loss"])

        mse = tf.reduce_mean(tf.square(y_test - mean_pred))

        return (
            mean_pred.numpy(),
            predictive_std.numpy(),
            nll_value,
            mse.numpy(),
        )