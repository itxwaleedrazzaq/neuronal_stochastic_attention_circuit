import tensorflow as tf
from neuronal_attention_circuit import NAC

@tf.keras.utils.register_keras_serializable(package="neuronal_attention_circuit", name="OUWrap")
class OUWrap(tf.keras.layers.Layer):
    """
    Neuronal Stochastic Attention Circuit (NSAC) Wrapper.

    This layer wraps NAC dynamics in Ornstein-Uhlenbeck (OU) dynamics.
    It models the attention process as a stochastic differential equation,
    providing both a predictive mean and a standard deviation (uncertainty).
    """

    def __init__(
        self,
        nac_instance: NAC,
        output_dim: int = 1,
        bn_mean: float = 0.0,
        bn_std: float = 0.1,
        activation=None,
        return_attention: bool = False,
        return_sequences: bool = False,
        **kwargs,
    ):
        """
        Args:
            nac_instance: Instance of the Neuronal Attention Circuit to wrap.
            output_dim: Dimension of the final probabilistic output.
            bn_mean: Mean for the Brownian noise in the OU process.
            bn_std: Standard deviation for the Brownian noise in the OU process.
            activation: Activation function for the hidden state.
            return_attention: Whether to return the attention weight matrix.
            return_sequences: Whether to return the full sequence or the last hidden state.
        """
        super().__init__(**kwargs)

        if not isinstance(nac_instance, NAC):
            raise ValueError("nac_instance must be an instance of NAC")
        
        self.nac = nac_instance
        self.output_dim = int(output_dim)
        self.bn_mean = float(bn_mean)
        self.bn_std = float(bn_std)
        self.activation = tf.keras.activations.get(activation)
        self.return_attention = bool(return_attention)
        self.return_sequences = bool(return_sequences)

        # NCP Projection to generate OU parameters: mu, theta, and sigma
        self.ncp_out = self.nac._make_inter_to_motor_projections("ou_out", output=3)

        # Standard attention output projection
        self.attention_out_proj = tf.keras.layers.Dense(
            self.nac.d_model * 2,
            use_bias=self.nac.use_bias,
            name="attention_out_projection"
        )

        # Probabilistic regression output
        self.mean_head = tf.keras.layers.Dense(
            self.output_dim,
            use_bias=self.nac.use_bias,
            name="probabilistic_mean"
        )

        self.std_head = tf.keras.layers.Dense(
            self.output_dim,
            use_bias=self.nac.use_bias,
            name="probabilistic_std"
        )

    def compute_phi_kappa_psi_time(self, q, k, t):
        """
        Computes the SDE parameters and time interpolation.
        
        Returns:
            phi: Equilibrium mean [B, H, Tq, K]
            kappa: Mean reversion speed [B, H, Tq, K]
            psi: Volatility (diffusion) [B, H, Tq, K]
            t_interp: Interpolated time steps [B, H, Tq, K]
            topk_idx: Indices of the selected sparse attention keys
        """
        batch_size, num_heads, seq_len_q = tf.shape(q)[0], tf.shape(q)[1], tf.shape(q)[2]

        # Extract sparse top-k pairwise interactions
        pair_features, topk_idx = self.nac.sparse_topk_pairwise(q, k, K=self.nac.topk)
        effective_k = tf.shape(pair_features)[3]

        # Generate OU parameters from the NCPCell
        flat_pairs = tf.reshape(pair_features, [-1, tf.shape(pair_features)[-1]])
        raw_params = self.ncp_out(tf.expand_dims(flat_pairs, axis=1))
        raw_params = tf.squeeze(raw_params, axis=1)

        # Reshape to parameter dimensions [Batch, Heads, Tq, K_eff, 3]
        param_tensor = tf.reshape(raw_params, [batch_size, num_heads, seq_len_q, effective_k, 3])
        
        phi = tf.nn.tanh(param_tensor[..., 0])
        kappa = tf.nn.softplus(param_tensor[..., 1]) + self.nac.tau_epsilon
        psi = tf.nn.softplus(param_tensor[..., 2]) + self.nac.tau_epsilon

        # Compute time-dependent gates
        time_gate_a = self.nac.time_a(pair_features)
        time_gate_b = self.nac.time_b(pair_features)

        # Handle time scalar or tensor logic
        if t is None:
            t_val = tf.cast(tf.constant(1.0), pair_features.dtype)
            t_expanded = tf.reshape(t_val, [1, 1, 1, 1, 1])
        else:
            t_val = tf.cast(t, pair_features.dtype)
            t_expanded = tf.reshape(t_val, [batch_size, 1, seq_len_q, 1, 1])
            
        t_interp = tf.nn.sigmoid(time_gate_a * t_expanded + time_gate_b)[..., 0]
        
        return phi, kappa, psi, t_interp, topk_idx
    
    def call(self, inputs, mask=None, training=None):
        """
        Forward computation solving the OU dynamics for stochastic attention.
        """
        if isinstance(mask, (list, tuple)):
            mask = mask[0] if len(mask) > 0 else None
            if mask is None or (hasattr(mask, 'shape') and mask.shape == ()):
                mask = None

        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                x, t = inputs
                q_in = k_in = v_in = x
            elif len(inputs) == 3:
                q_in, k_in, v_in = inputs
                t = None
            elif len(inputs) == 4:
                q_in, k_in, v_in, t = inputs
            else:
                raise ValueError("Unsupported input tuple length")
        else:
            q_in = k_in = v_in = inputs
            t = None

        # Project through NAC's Sensory Gate
        q = self.nac.q_proj(q_in)
        k = self.nac.k_proj(k_in)
        v = self.nac.v_proj(v_in)

        # Multi-Head NAC's Splitting
        qh = self.nac.split_heads(q)
        kh = self.nac.split_heads(k)
        vh = self.nac.split_heads(v)

        # Compute OU Parameters 
        phi, kappa, psi, dt, topk_idx = self.compute_phi_kappa_psi_time(qh, kh, t)

        # OU Mean and Variance
        # Mean: phi * (1 - exp(-kappa * t))
        # Variance: (psi^2 / 2*kappa) * (1 - exp(-2 * kappa * t))
        ou_mean = phi * (1.0 - tf.exp(-kappa * dt))
        ou_var = (psi ** 2) * (1.0 - tf.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
        ou_stddev = tf.sqrt(tf.maximum(ou_var, 1e-9))

        # Brownian Motion Realization
        noise = ou_stddev * tf.random.normal(mean=self.bn_mean, stddev=self.bn_std, 
                                             shape=tf.shape(ou_stddev), dtype=ou_stddev.dtype)
            
        attn_logits = ou_mean + noise

        # Apply Sparse Masking
        if mask is not None:
            mask = tf.cast(mask, attn_logits.dtype)
            mask_exp = tf.expand_dims(tf.expand_dims(mask, 1), 1)
            mask_gathered = tf.gather(mask_exp, topk_idx, batch_dims=2, axis=2)
            attn_logits = attn_logits * mask_gathered

        # logistic-normalization distribution
        attn_weights = tf.nn.softmax(attn_logits)
        attn_weights = self.nac.attn_dropout(attn_weights, training=training)

        # head specific output computation 
        vh_topk = tf.gather(vh, topk_idx, batch_dims=2, axis=2)
        weighted = attn_weights[..., tf.newaxis] * vh_topk
        head_out = tf.reduce_sum(weighted, axis=3) 

        # multihead extension with double dimension projections
        combined = self.nac.combine_heads(head_out)
        projected = self.attention_out_proj(combined)
        
        if self.activation is not None:
            projected = self.activation(projected)

        # Split into predictive and uncertainty streams
        features_mean, features_std = tf.split(projected, 2, axis=-1)

        final_mean = self.mean_head(features_mean)
        final_std = self.std_head(features_std)

        # Sequence Handling
        if not self.return_sequences:
            final_mean = final_mean[:, -1, :]
            final_std = final_std[:, -1, :]

        return final_mean, final_std

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "activation": tf.keras.activations.serialize(self.activation),
            "bn_mean": self.bn_mean,
            "bn_std": self.bn_std,
            "return_attention": self.return_attention,
            "return_sequences": self.return_sequences,
        })
        return config