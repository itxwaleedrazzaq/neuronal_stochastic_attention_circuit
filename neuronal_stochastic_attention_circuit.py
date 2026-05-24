import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="neuronal_stochastic_attention_circuit",name="NSAC")
class NSAC(tf.keras.Model):
    """
    Neuronal Stochastic Attention Circuit (NSAC)
 
    Uses internal Brownian noise realizations to quantify uncertainty.
    - Epistemic: Variance across different stochastic forward passes.
    - Aleatoric: Mean of the predicted internal circuit noise (std^2).
    """

    def __init__(
        self,
        stochastic_model_fn,
        mc_samples: int = 20,
        ood_mean: float = 0.0,
        ood_std: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        '''
        Args:
            stochastic_model_fn: OUWrapped NAC Model that implements the core stochastic circuit with Brownian noise. 
            mc_samples : No. of Monte Carlo samples during training and inference.
            lambda_reg: Weight for the OOD-based Epistemic regularization term in the loss.
            ood_std: Standard deviation for generating OOD samples via Gaussian perturbation during training.
        '''  

        self.stochastic_circuit = stochastic_model_fn
        self.mc_samples = mc_samples
        self.ood_mean = ood_mean
        self.ood_std = ood_std
        self.build((None, stochastic_model_fn.input_shape[-1]))

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

        # Generate OOD samples via Gaussian perturbation for regularization
        x_ood = x_id + tf.random.normal(
            shape=tf.shape(x_id),
            mean=self.ood_mean,
            stddev=self.ood_std,
            dtype=x_id.dtype
        )

        with tf.GradientTape() as tape:
            id_samples = []
            ood_samples = []
             #MC sampling
            for _ in range(self.mc_samples):
                mu_id, std_id = self.stochastic_circuit(x_id, training=True)
                mu_ood, std_ood = self.stochastic_circuit(x_ood, training=True)

                id_samples.append((mu_id, std_id))
                ood_samples.append((mu_ood, std_ood))
            total_loss, nll_loss, reg_loss = self.loss_fn(y_id, [id_samples, ood_samples])      

        gradients = tape.gradient(total_loss, self.stochastic_circuit.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.stochastic_circuit.trainable_variables))

        return {"loss": total_loss, "nll": nll_loss, "reg": reg_loss}

    def test_step(self, data):
        x, y = data
        x_id = tf.cast(x, tf.float32)
        y_id = tf.cast(y, tf.float32)
        id_samples = []
        ood_samples = []

        x_ood = x_id + tf.random.normal(
            shape=tf.shape(x_id),
            mean=self.ood_mean,
            stddev=self.ood_std,
            dtype=x_id.dtype
        )

        for _ in range(self.mc_samples):
            mu_id, std_id = self.stochastic_circuit(x_id, training=True)
            mu_ood, std_ood = self.stochastic_circuit(x_ood, training=True)

            id_samples.append((mu_id, std_id))
            ood_samples.append((mu_ood, std_ood))
        total_loss, nll_loss, reg_loss = self.loss_fn(y_id, [id_samples, ood_samples])    

        return {"loss": total_loss, "nll": nll_loss, "reg": reg_loss }

    def call(self, inputs, training=None):
        return self.stochastic_circuit(inputs, training=training)
    
    def summary(self):
        return self.stochastic_circuit.summary()  

    def _get_stochastic_components(self, x):
        """
        Internal helper to run multiple Brownian noise realizations and 
        calculate the variance decomposition.
        """
        means = []
        variances = []

        for _ in range(self.mc_samples):
            m, s  = self.stochastic_circuit(x, training=False) 
            means.append(m)
            variances.append(tf.square(tf.exp(s))) # std^2 = aleatoric variance

        means = tf.stack(means)
        variances = tf.stack(variances)

        # Decomposition: Total = Var(Means) + Mean(Vars)
        mean_prediction = tf.reduce_mean(means, axis=0)
        epistemic = tf.math.reduce_variance(means, axis=0)
        aleatoric = tf.reduce_mean(variances, axis=0)

        return mean_prediction, aleatoric, epistemic

    def predict(self, x, **kwargs):
        mean, aleatoric, epistemic = self._get_stochastic_components(x)
        total_uncertainty = epistemic + aleatoric
        return mean.numpy(), total_uncertainty.numpy()


    def predict_with_uncertainty(self, x, **kwargs):
        mean, aleatoric, epistemic = self._get_stochastic_components(x)        
        return mean.numpy(), aleatoric.numpy(), epistemic.numpy()

    def comprehensive_evaluate(self, X_test, y_test):
        X_test = tf.cast(X_test, tf.float32)
        y_test = tf.cast(y_test, tf.float32)

        mu, log_std = self.stochastic_circuit(X_test, training=False)
        std = tf.exp(log_std)

        mse_value = tf.reduce_mean(tf.square(y_test - mu))

        results = super().evaluate(
            X_test,
            y_test,
            return_dict=True,
            verbose=0
        )
        nll_value = float(results["nll"])

        # Return full arrays for metrics
        return mu.numpy(), std.numpy(), nll_value, mse_value.numpy()

    def get_config(self):
        config = super().get_config()
        config.update({
            'stochastic_model_fn': self.stochastic_circuit,
            'mc_samples': self.mc_samples,
            'ood_mean': self.ood_mean,
            'ood_std': self.ood_std,
        })
        return config
    
    def save(self, filepath, **kwargs):
        self.stochastic_circuit.save(filepath, **kwargs)

    def save_weights(self, filepath, **kwargs):
        self.stochastic_circuit.save_weights(filepath, **kwargs)

    def load_weights(self, filepath, **kwargs):
        self.stochastic_circuit.load_weights(filepath, **kwargs)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
