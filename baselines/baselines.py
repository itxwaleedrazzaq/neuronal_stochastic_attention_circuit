import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable(package="neuronal_stochastic_attention_circuit", name="Gaussian-MLE")
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
        if optimizer:
            self.optimizer = optimizer
        if loss:
            self.loss_fn = loss
        super().compile(metrics=metrics, **kwargs)

    def train_step(self, data):
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

        mu, log_std = self.model(X_test, training=False)
        std = tf.exp(log_std)

        mu_mean = tf.reduce_mean(mu).numpy()
        std_mean = tf.reduce_mean(std).numpy()
        mse_value = tf.reduce_mean(tf.square(y_test - mu)).numpy()

        results = super().evaluate(
            X_test,
            y_test,
            return_dict=True,
            verbose=0
        )

        nll_value = float(results["nll"])

        return mu_mean, std_mean, nll_value, mse_value

    def get_config(self):
        config = super().get_config()
        config.update({
            "model": tf.keras.utils.serialize_keras_object(self.model),
            "model_name": self.model_name,
        })
        return config
    @classmethod
    def from_config(cls, config):
        model = tf.keras.utils.deserialize_keras_object(config.pop("model"))
        return cls(model_fn=model, **config)

@tf.keras.utils.register_keras_serializable(package="neuronal_stochastic_attention_circuit", name="DeepEnsemble")
class DeepEnsemble(tf.keras.Model):
    def __init__(self, model_fn, num_models=20, model_name='DeepEnsemble'):
        super().__init__()
        self.model = model_fn
        self.model_name = model_name
        self.num_models = num_models

        self.build((None, model_fn.input_shape[-1]))

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)
        return self.model(inputs, training=training)

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        if optimizer:
            self.optimizer = optimizer
        if loss:
            self.loss_fn = loss
        super().compile(metrics=metrics, **kwargs)

    def train_step(self, data):
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


    def calculate_uncertainty(self, X, noise_scale=0.001):
        X = tf.cast(X, tf.float32)
        base_weights = self.model.get_weights()
        preds = []
        log_stds = []

        for _ in range(self.num_models):
            perturbed_weights = []
            for w in base_weights:
                scale = np.std(w) if np.std(w) > 0 else 1.0
                noise = np.random.normal(loc=0.0, scale=noise_scale * scale, size=w.shape)
                perturbed_weights.append(w + noise)

            self.model.set_weights(perturbed_weights)
            mu, log_std = self.model(X, training=False)
            mu = mu.numpy()
            log_std = log_std.numpy()
            preds.append(mu)
            log_stds.append(log_std)

        self.model.set_weights(base_weights)
        all_preds = np.stack(preds, axis=0)
        all_log_stds = np.stack(log_stds, axis=0)
        mean_pred = np.mean(all_preds, axis=0)
        epistemic_unc = np.var(all_preds, axis=0)
        aleatoric_unc = np.mean(np.exp(2 * all_log_stds), axis=0)
        total_uncertainty = epistemic_unc + aleatoric_unc
        return mean_pred, aleatoric_unc, epistemic_unc, total_uncertainty

    def predict_with_uncertainty(self, X, noise_scale=0.01):
        mean_pred, aleatoric, epistemic, _ = self.calculate_uncertainty(X, noise_scale=noise_scale)
        return mean_pred, aleatoric, epistemic
    
    def predict(self, X, *kwargs):
        mean_pred, epistemic, aleatoric, total_uncertainty = self.calculate_uncertainty(X)
        return mean_pred, total_uncertainty

    def comprehensive_evaluate(self, X_test, y_test):
        X_test = tf.cast(X_test, tf.float32)
        y_test = tf.cast(y_test, tf.float32)

        mean_pred, aleatoric, epistemic, total_uncertainty = self.calculate_uncertainty(X_test)

        variance = tf.clip_by_value(total_uncertainty, 1e-6, 1e6)
        results = super().evaluate(
            X_test,
            y_test,
            return_dict=True,
            verbose=0
        )
        nll = float(results["nll"])
        mse = tf.reduce_mean(tf.square(y_test - mean_pred))
        predictive_std = tf.sqrt(variance)
        return (
            mean_pred,
            predictive_std,
            nll,
            mse.numpy(),
        )
    def get_config(self):
        config = super().get_config()
        config.update({
            "model": tf.keras.utils.serialize_keras_object(self.model),
            "num_models": self.num_models,
            "model_name": self.model_name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        model = tf.keras.utils.deserialize_keras_object(config.pop("model"))
        return cls(model_fn=model, **config)



@tf.keras.utils.register_keras_serializable(package="neuronal_stochastic_attention_circuit", name="MC_Dropout")
class MC_dropout(tf.keras.Model):
    def __init__(self, model_fn, mc_samples=20, model_name='MC_Dropout'):
        super().__init__()
        self.model = model_fn
        self.model_name = model_name
        self.mc_samples = mc_samples

        self.build((None, model_fn.input_shape[-1]))

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)
        return self.model(inputs, training=training)

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        if optimizer:
            self.optimizer = optimizer
        if loss:
            self.loss_fn = loss
        super().compile(metrics=metrics, **kwargs)


    def train_step(self, data):
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
        means = []
        variances = []
        for _ in range(self.mc_samples):
            mean, log_std = self.model(X, training=True)
            var = tf.nn.softplus(2.0 * log_std) + 1e-6  
            means.append(mean)
            variances.append(var)
        means = np.stack(means, axis=0)          # [T, N, 1]
        variances = np.stack(variances, axis=0)  # [T, N, 1]
        mean_pred = np.mean(means, axis=0)
        epistemic_unc = np.var(means, axis=0)
        aleatoric_unc = np.mean(variances, axis=0)
        total_uncertainty = epistemic_unc + aleatoric_unc
        return mean_pred, aleatoric_unc, epistemic_unc, total_uncertainty

    # Simple Prediction API
    def predict_with_uncertainty(self, X):
        mean_pred, aleatoric, epistemic, _ = self.calculate_uncertainty(X)
        return mean_pred, aleatoric, epistemic

    # Evaluation Metrics
    def comprehensive_evaluate(self, X_test, y_test):
        X_test = tf.cast(X_test, tf.float32)
        y_test = tf.cast(y_test, tf.float32)

        mean_pred, aleatoric, epistemic, total_uncertainty = self.calculate_uncertainty(X_test)

        variance = tf.clip_by_value(total_uncertainty, 1e-6, 1e6)
        results = super().evaluate(
            X_test,
            y_test,
            return_dict=True,
            verbose=0
        )
        nll = float(results["nll"])
        mse = tf.reduce_mean(tf.square(y_test - mean_pred))
        predictive_std = tf.sqrt(variance)
        return (
            mean_pred,
            predictive_std.numpy(),
            nll,
            mse.numpy(),
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "model": tf.keras.utils.serialize_keras_object(self.model),
            "mc_samples": self.mc_samples,
            "model_name": self.model_name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        model = tf.keras.utils.deserialize_keras_object(config.pop("model"))
        return cls(model_fn=model, **config)
    
@tf.keras.utils.register_keras_serializable(package="neuronal_stochastic_attention_circuit", name="Evidential")
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
    
    def predict(self, X):
        mu , aleatoric, epistemic = self.calculate_uncertainty(X)
        total_uncertainty = aleatoric + epistemic
        return mu.numpy(), total_uncertainty.numpy()

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

        mean_pred, aleatoric, epistemic = self.calculate_uncertainty(X_test)

        total_variance = aleatoric + epistemic
        mean_pred = tf.reduce_mean(mean_pred)
        total_variance = tf.reduce_mean(total_variance)

        predictive_std = tf.sqrt(total_variance) + 1e-8

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "model": tf.keras.utils.serialize_keras_object(self.model),
            "lambda_reg": self.lambda_reg,
            "model_name": self.model_name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        model = tf.keras.utils.deserialize_keras_object(config.pop("model"))
        return cls(model_fn=model, **config)
    
@tf.keras.utils.register_keras_serializable(package="neuronal_stochastic_attention_circuit", name="SDE_Net")
class SDE_Net(tf.keras.Model):
    def __init__(self, input_dim, output_dim, layer_depth=6, hidden_dim=50, mc_samples=20, model_name='SDE_Net'):
        super().__init__()
        self.model_name  = model_name
        self.layer_depth = layer_depth
        self.mc_samples  = mc_samples
        self.sigma_max   = 0.1
        self.deltat      = 4.0 / layer_depth

        self.downsampling_layers = tf.keras.layers.Dense(hidden_dim)
        self.drift_fc = tf.keras.layers.Dense(hidden_dim)
        self.diffusion_fc1 = tf.keras.layers.Dense( hidden_dim * 2)
        self.diffusion_fc2 = tf.keras.layers.Dense(1)
        self.output_relu  = tf.keras.layers.Activation('sigmoid')
        self.output_dense = tf.keras.layers.Dense(output_dim * 2)

        self(tf.zeros((1, input_dim), dtype=tf.float32))

    def _drift(self, x):
        return self.drift_fc(x)

    def _ensure_2d(self, x):
        x = tf.cast(x, tf.float32)
        if len(x.shape) > 2:
            x = tf.reshape(x, (tf.shape(x)[0], -1))
        elif len(x.shape) == 1:
            x = tf.expand_dims(x, axis=-1)
        return x

    def _diffusion(self, x):
        out = self.diffusion_fc1(x)
        out = self.diffusion_fc2(out)
        return tf.nn.sigmoid(out)                    

    def call(self, inputs, training_diffusion=False):
        inputs = self._ensure_2d(inputs)

        out = self.downsampling_layers(inputs)   

        if not training_diffusion:
            diffusion_term = tf.cast(self.sigma_max, tf.float32) * self._diffusion(out)

            for _ in range(self.layer_depth):
                out = (
                    out
                    + self._drift(out) * tf.cast(self.deltat, tf.float32)
                    + diffusion_term
                    * tf.cast(tf.math.sqrt(self.deltat), tf.float32)
                    * tf.random.normal(tf.shape(out), dtype=tf.float32)
                )

            fc_out = self.output_relu(out)
            fc_out = self.output_dense(fc_out)
            mean, sigma = tf.split(fc_out, num_or_size_splits=2, axis=-1)
            return mean, sigma+1e-6

        else:
            return self._diffusion(out)   

    def compile(self, optimizer_drift=None, optimizer_fc=None, optimizer_dsl=None, optimizer_diffusion=None,loss=None, metrics=None, **kwargs):
        if optimizer_drift:
            self.optimizer_drift = optimizer_drift
        if optimizer_fc:
            self.optimizer_fc = optimizer_fc
        if optimizer_dsl:
            self.optimizer_dsl = optimizer_dsl
        if optimizer_diffusion:
            self.optimizer_diffusion = optimizer_diffusion
        if loss:
            self.loss_fn = loss                    
        super().compile(metrics=metrics, **kwargs)

    def train_step(self, data):
        x, y = data
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            mean, sigma = self.call(x, training_diffusion=False)
            nll = self.loss_fn(y, [mean, sigma])

        drift_grad= [tf.clip_by_norm(g, 100) for g in tape.gradient(nll, self.drift_fc.trainable_variables)]
        dsl_grad= [tf.clip_by_norm(g, 100) for g in tape.gradient(nll, self.downsampling_layers.trainable_variables)]
        fc_grad= [tf.clip_by_norm(g, 100) for g in tape.gradient(nll, self.output_dense.trainable_variables)]
        del tape

        self.optimizer_drift.apply_gradients(zip(drift_grad, self.drift_fc.trainable_variables))
        self.optimizer_dsl.apply_gradients(zip(dsl_grad,self.downsampling_layers.trainable_variables))
        self.optimizer_fc.apply_gradients(zip(fc_grad,self.output_dense.trainable_variables))

        diff_vars  = (self.diffusion_fc1.trainable_variables + self.diffusion_fc2.trainable_variables)
        real_label = tf.constant(0.0, dtype=tf.float32)
        fake_label = tf.constant(1.0, dtype=tf.float32)

        with tf.GradientTape() as real_tape:
            real_y    = tf.fill(tf.shape(self.call(x, training_diffusion=True)), real_label)
            real_pred = self.call(x, training_diffusion=True)
            real_loss = tf.reduce_mean(tf.square(real_y - real_pred))
        diff_grad1 = [tf.clip_by_norm(g, 100) for g in real_tape.gradient(real_loss, diff_vars)]

        with tf.GradientTape() as fake_tape:
            fake_x    = x + tf.cast(tf.random.normal(tf.shape(x), stddev=5), tf.float32)
            fake_y    = tf.fill(tf.shape(self.call(fake_x, training_diffusion=True)), fake_label)
            fake_pred = self.call(fake_x, training_diffusion=True)
            fake_loss = tf.reduce_mean(tf.square(fake_y - fake_pred))
        diff_grad2 = [tf.clip_by_norm(g, 100) for g in fake_tape.gradient(fake_loss, diff_vars)]

        combined = [g1 + g2 for g1, g2 in zip(diff_grad1, diff_grad2)]
        self.optimizer_diffusion.apply_gradients(zip(combined, diff_vars))
        return { "loss": nll, "nll": nll,}

    def test_step(self, data):
        x, y = data
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        mean, sigma = self.call(x, training_diffusion=False)
        loss = self.loss_fn(y, [mean, sigma])
        return {"loss": loss, "nll": loss}

    def calculate_uncertainty(self, X):
        X = tf.cast(X, tf.float32)
        means, variances = [], []

        for _ in range(self.mc_samples):
            mean, sigma = self.call(X, training_diffusion=False)
            means.append(mean.numpy())
            variances.append((sigma ** 2).numpy())

        means = np.stack(means, axis=0)         # [T, N, 1]
        variances = np.stack(variances, axis=0)         # [T, N, 1]

        mean_pred = np.mean(means, axis=0)
        epistemic_unc = np.var(means, axis=0)
        aleatoric_unc = np.mean(variances, axis=0)
        total_uncertainty = epistemic_unc + aleatoric_unc

        return mean_pred, aleatoric_unc, epistemic_unc, total_uncertainty

    def predict(self, X):
        mean_pred, aleatoric, epistemic, _ = self.calculate_uncertainty(X)
        return mean_pred, aleatoric + epistemic

    def predict_with_uncertainty(self, X):
        mean_pred, aleatoric, epistemic, _ = self.calculate_uncertainty(X)
        return mean_pred, aleatoric, epistemic

    def comprehensive_evaluate(self, X_test, y_test):
        X_test = tf.cast(X_test, tf.float32)
        y_test = tf.cast(y_test, tf.float32)

        mean_pred, aleatoric, epistemic, total_uncertainty = self.calculate_uncertainty(X_test)

        variance = tf.clip_by_value(total_uncertainty, 1e-6, 1e6)
        results = super().evaluate(
            X_test,
            y_test,
            return_dict=True,
            verbose=0
        )
        nll = float(results["loss"])
        mse = tf.reduce_mean(tf.square(y_test - mean_pred))
        predictive_std = tf.sqrt(variance)

        return (mean_pred, predictive_std.numpy(), nll, mse.numpy())

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": None,  # explained below
            "output_dim": self.output_dense.units // 2,
            "layer_depth": self.layer_depth,
            "hidden_dim": self.downsampling_layers.units,
            "mc_samples": self.mc_samples,
            "model_name": self.model_name,
        })
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)