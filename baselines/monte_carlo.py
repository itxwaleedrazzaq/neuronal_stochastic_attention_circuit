import tensorflow as tf
import numpy as np

class MC_dropout(tf.keras.Model):
    def __init__(self, model_fn, model_name='MC_Dropout'):
        super().__init__()
        self.model = model_fn
        self.model_fn = model_fn
        self.model_name = model_name

        self.build((None, model_fn.input_shape[-1]))

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)
        return self.model(inputs, training=training)

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)


    def calculate_uncertainty(self, X, num_samples=5,min_dropout=0.0, max_dropout=0.005):
        X = tf.cast(X, tf.float32)
        base_weights = self.model.get_weights()
        preds = []

        for _ in range(num_samples):
            new_dropout = np.random.uniform(min_dropout, max_dropout)
            def clone_fn(layer):
                if isinstance(layer, tf.keras.layers.Dropout):
                    return tf.keras.layers.Dropout(rate=new_dropout)
                return layer.__class__.from_config(layer.get_config())
            temp_model = tf.keras.models.clone_model(self.model,clone_function=clone_fn)
            temp_model.set_weights(base_weights)
            prediction = temp_model(X, training=True).numpy()
            preds.append(prediction)
        all_samples = np.stack(preds, axis=0)
        mean_pred = np.mean(all_samples, axis=0)
        epistemic_unc = np.var(all_samples, axis=0)
        aleatoric_unc = np.zeros_like(epistemic_unc)
        total_uncertainty = epistemic_unc + aleatoric_unc

        return mean_pred, aleatoric_unc, epistemic_unc, total_uncertainty

    def predict_with_uncertainty(self, X, num_samples=20):
        mean_pred, epistemic, aleatoric, total_uncertainty = self.calculate_uncertainty(X, num_samples)
        return mean_pred, aleatoric, epistemic
    

    def comprehensive_evaluate(self, X_test, y_test, n_samples=10):

        X_test = tf.cast(X_test, tf.float32)
        y_test = tf.cast(y_test, tf.float32)

        # --- Uncertainty Decomposition ---
        mean_pred, aleatoric, epistemic, total_uncertainty = self.calculate_uncertainty(X_test)

        total_variance = aleatoric + epistemic
        mean_pred = tf.reduce_mean(mean_pred)
        total_variance = tf.reduce_mean(total_variance)


        predictive_std = tf.sqrt(tf.reduce_mean(total_variance) + 1e-8)

        nll = 0.5 * tf.reduce_min(
            tf.math.log(2.0 * tf.constant(tf.constant(3.141592653589793), dtype=tf.float32) * total_variance + 1e-8)
            + tf.square(y_test - mean_pred) / (total_variance + 1e-8)
        )

        mse = tf.reduce_mean(tf.square(y_test - mean_pred))

        return (
            mean_pred.numpy(),
            predictive_std.numpy(),
            nll.numpy(),
            mse.numpy(),
        )